import sys
import numpy as np
import jsonlines
from statistics import mean

from scipy.optimize import linear_sum_assignment


class ConceptBestMatch():
    def __init__(self, input_data, words_sos=False, words_eos=True, concepts_sos= False, concepts_eos = False):
        self.msg_loc = 0
        self.phrase_loc = 1
        self.words_sos = words_sos
        self.words_eos = words_eos
        self.concepts_sos = concepts_eos
        self.concepts_eos = concepts_sos
        assert len(input_data) > 0, f'Got empty input data to evaluate'
        if isinstance(input_data[0], dict):
            self.input_data = [list(sample.values()) for sample in input_data]
        else:
            self.input_data = input_data

    # Remove start_of_seq/end_of_seq if communication has such
    def adjust_sos_eos(self, seq, sos, eos):
        if sos:
            seq = seq[1:]
        if eos:
            seq = seq[:-1]
        assert len(seq) > 0, f'Seq becomes zero after sos/eos adjustment'
        return seq

    def extract_data(self, input_data):
        word2idx = []
        cpt2idx = []
        counter = {}
        for sample in input_data:
            message = sample[self.msg_loc]
            phrase = sample[self.phrase_loc]
            words = message.split(".")
            concepts = phrase.split(".")
            # We add multiple (word*concept) edges - each for each (word-concept) pair
            # We count the edges again after having the best_match assignment
            for word in self.adjust_sos_eos(words, self.words_sos, self.words_eos):
                 for concept in self.adjust_sos_eos(concepts, self.concepts_sos, self.concepts_eos):
                    if word not in word2idx: word2idx.append(word)
                    if concept not in cpt2idx: cpt2idx.append(concept)
                    if word not in counter.keys():
                        counter[word] = {}
                    if concept not in counter[word].keys():
                        counter[word][concept] = 0
                    counter[word][concept] += 1
        return word2idx, cpt2idx, counter

    def calc_match(self, word2idx, cpt2idx, counter):
        # Create a cost matrix where each element represents the match score
        cost_matrix = np.zeros((len(word2idx), len(cpt2idx)))
        for i, word in enumerate(word2idx):
            for j, concept in enumerate(cpt2idx):
                cost_matrix[i, j] = 0
                if word in counter.keys() and concept in counter[word].keys():
                    # We are adding negative values as optimization is done on the cost (lower is better)
                    cost_matrix[i, j] = -counter[word][concept]

        # Use linear sum assignment to find the best match
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        return row_indices, col_indices, cost_matrix

    def report_results(self, input_data, word2idx, cpt2idx, row_indices, col_indices, cost_matrix):
        total_edges = 0
        good_edges = 0
        ambiguous_edges = 0
        paraphrase_edges = 0
        unmatched_concepts = 0
        total_concepts = 0
        unique_messages = set()
        unique_phrases = set()
        precision_list = []
        recall_list = []
        unique_words = set()
        bm_word2cpt = {}
        for word_idx, cpt_idx in zip(row_indices, col_indices):
            if cost_matrix[word_idx, cpt_idx] == 0: continue
            bm_word2cpt[word2idx[word_idx]] = cpt2idx[cpt_idx]

        for sample in input_data:
            words = sample[self.msg_loc].split(".")
            words = self.adjust_sos_eos(words, self.words_sos, self.words_eos)
            unique_words.update(words)
            unique_messages.add(".".join(words))

            concepts = sample[self.phrase_loc].split(".")
            concepts = self.adjust_sos_eos(concepts, self.concepts_sos, self.concepts_eos)
            unique_phrases.add(".".join(concepts))

            # For the normalization
            total_edges += max(len(words), len(concepts))

            good_edges_in_msg = set() # don't count duplicates
            for word in words:
                if word in bm_word2cpt.keys():
                    matched_concept = bm_word2cpt[word]
                    if matched_concept in concepts:
                        good_edges += 1
                        good_edges_in_msg.add(word)
                    else:
                        ambiguous_edges += 1
                else:
                    paraphrase_edges += 1

            precision_list.append(len(good_edges_in_msg)/len(words))
            recall_list.append(len(good_edges_in_msg)/len(concepts))

            #Count unmatched concepts
            for concept in concepts:
                cpt_idx = cpt2idx.index(concept)
                total_concepts += 1
                if cpt_idx not in col_indices:
                    unmatched_concepts += 1

        assert len(unique_words) == len(word2idx)
        results = {"Unique_messages": len(unique_messages), "Unique_words": len(word2idx),
                    "Unique_phrases": len(unique_phrases), "Unique_concepts": len(cpt2idx),
                    # This cannot be calculated with the addition of conditional edges
                    "Unique_edges": -1, # np.count_nonzero(cost_matrix), # This cannot be calculated with the
                    "Total_edges": total_edges,
                    "Good_edges": good_edges, "Ambiguous_edges": ambiguous_edges,
                    "Paraphrase_edges": paraphrase_edges, "Total_concepts": total_concepts,
                    "Unmatched_concepts": unmatched_concepts,
                    "Match_score": f'{(good_edges / total_edges):.3f}',
                     # Consider the use of the term polysemantic
                    "Ambiguous_score": f'{(ambiguous_edges / total_edges):.3f}',
                    "Paraphrase_score": f'{(paraphrase_edges / total_edges):.3f}',
                    "Unmatched_score": f'{(unmatched_concepts / total_concepts):.3f}',
                    "Precision": f'{mean(precision_list):.3f}', "Recall": f'{mean(recall_list):.3f}',
                    "Match_mapping": [(word2idx[i], cpt2idx[j], -cost_matrix[i,j]) for i, j in zip(row_indices, col_indices)]
                   }
        return results

    def calc_best_match(self):
        word2idx, cpt2idx, edges_counter = self.extract_data(self.input_data)
        row_indices, col_indices, cost_matrix = self.calc_match(word2idx, cpt2idx, edges_counter)
        results = self.report_results(self.input_data, word2idx, cpt2idx, row_indices, col_indices, cost_matrix)
        return results

if __name__ == '__main__':

    # input_file = sys.argv[1]
    # input_file = "gs_rnn_1_16_16_10.test.jsonl"
    input_file = "comm_results.jsonl"
    words_eos = False if len(sys.argv) > 2 else True
    with jsonlines.open(input_file, 'r') as f:
        input_data_from_file = [line for line in f]
    cbm = ConceptBestMatch(input_data_from_file, words_eos=words_eos)
    results = cbm.calc_best_match()
    print(results)

