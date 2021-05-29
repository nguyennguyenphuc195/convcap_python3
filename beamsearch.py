import torch 
import torch.nn as nn
import itertools
import numpy as np

class beamsearch:
    """Beam search on output softmax distribution"""
    def __init__(self, beam_size, batch_size, maxlen):
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.current_beam_size = [self.beam_size for i in range(self.batch_size)]
        self.maxlen = maxlen
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.done_beams = [[] for i in range(self.batch_size)]
        self.beams      = [[{"words" : [0], "total_logprob" : 0.0}] for i in range(self.batch_size)]

        self.first_pass = True

    def reset(self):
        self.first_pass = True
    
    def get_start_end_indices(self):
        """
        Get array of indices 
        [
            index of 1st beam of 1st sample/picture,
            index of 1st beam of 2nd sample/picture,
            index of 1st beam of 3rd sample/picture,
            index of 1st beam of 4th sample/picture,
            ...
        ]
        """
        start_end_idx = [0]
        start_end_idx.extend(self.current_beam_size)
        start_end_idx = torch.cumsum(torch.LongTensor(start_end_idx), 0)
        self.idx = start_end_idx

    def expand_beam(self, outputs):
        selected_word_indices = []
        selected_beam_indices = []
        self.get_start_end_indices()

        for i in range(self.batch_size):
            start_idx, end_idx = self.idx[i], self.idx[i + 1]

            if start_idx == end_idx:
                assert (len(self.done_beams[i]) > 0)
                continue
            
            output = outputs[start_idx : end_idx]
            beam_word_logprobs = self.logsoftmax(output).cpu().data.tolist()

            #Get candidate beams
            candidate_beams = []
            app = candidate_beams.append # Give speedup ?

            if self.first_pass:
                beam_word_logprobs = [beam_word_logprobs[0]]

            for source, word_logprobs in enumerate(beam_word_logprobs):
                old_beam = self.beams[i][source]

                sorted_probs    = sorted(enumerate(word_logprobs), key = lambda x : x[1], reverse=True)
                sorted_word_idx = [item[0] for item in sorted_probs]

                for word_idx in sorted_word_idx[:self.beam_size]:
                    logprob = word_logprobs[word_idx]

                    words = list(old_beam["words"])
                    words.append(word_idx)
                    app({
                        "source": source, "logprob": logprob, "last_word_idx" : word_idx,
                        "words" : words, "total_logprob" : logprob + old_beam["total_logprob"]
                    })
            candidate_beams = sorted(candidate_beams, key= lambda x : x["total_logprob"], reverse=True)
            self.beams[i] = candidate_beams[:self.beam_size]

            #move every beam with <EOS> as last word to done_beams, else to remaining_beams
            remaining_beams = []
            for beam in self.beams[i]:
                if beam["last_word_idx"] == 0:
                    self.done_beams[i].append(beam)
                else:
                    remaining_beams.append(beam)
            self.beams[i] = remaining_beams

            end_idx = start_idx + len(self.beams[i])
            self.current_beam_size[i] = end_idx - start_idx
            if (len(self.beams[i]) == 0):
                continue

            beam_indices = [start_idx + item['source'] for item in self.beams[i]]
            word_indices = [item["last_word_idx"] for item in self.beams[i]]
            selected_beam_indices.extend(beam_indices)
            selected_word_indices.extend(word_indices)
        if self.first_pass:
            self.first_pass = False

        return selected_beam_indices, selected_word_indices

    def get_result(self):
        generated_caption = torch.zeros(self.maxlen, self.batch_size, dtype=torch.long)
        for i in range(self.batch_size):
            if len(self.done_beams[i]) == 0:
                self.done_beams[i] = self.beams[i]
            self.done_beams[i] = sorted(self.done_beams[i], key=lambda x : x["total_logprob"], reverse=True)
            best_beam = self.done_beams[i][0]["words"][1:]

            while len(best_beam) < self.maxlen:
                best_beam.append(0)

            best_beam = best_beam[:self.maxlen]
            generated_caption[:, i] = torch.LongTensor(best_beam)
        
        return generated_caption

