import numpy as np


class HMMTagger:
    def __init__(self, states, observations):
        """Initialize HMM model with states and observations
        :param states: array with unique hidden states (list)
        :param observations: array with unique observations (list)
        """
        # add 'Unk' to handle unkown tokens
        self.states = states
        self.observations = [*observations, 'Unk']
        self.states_num = len(self.states)
        self.observations_num = len(self.observations)

        self.init_prob = np.zeros(shape=(1, self.states_num))
        self.transition_matrix = np.zeros(shape=(self.states_num, self.states_num))
        self.emission_matrix = np.zeros(shape=(self.states_num, self.observations_num))

        self.states_to_idx = {state: idx for idx, state in enumerate(self.states)}
        self.observations_to_idx = {obs: idx for idx, obs in enumerate(self.observations)}
        self.inverse_idx_states = {idx: obs for obs, idx in self.states_to_idx.items()}

    def fit(self, train_data):
        """Estimate initial probability vector, transition and emission matrices
        :param train_data: list of sentecnes where each sentence is represented by list of tuples (state, observation)

        Example: train_data = [[('O', 'Thousands'),
                                ('O', 'of'),
                                ('O', 'demonstrators'),
                                ('O', 'have'),
                                ('O', 'marched'),
                                ('O', 'through'),
                                ('B-geo', 'London')],
                               [('B-gpe', 'Iranian'),
                                ('O', 'officials'),
                                ('O', 'say'),
                                ('O', 'they'),
                                ('O', 'expect'),
                                ('O', 'to'),
                                ('O', 'get'),
                                ('O', 'access')]]

        """
        final_counts = np.zeros(self.states_num)

        for elem in train_data:
            self.init_prob[0, self.states_to_idx[elem[0][0]]] += 1

            for i in range(1, len(elem)):
                self.transition_matrix[self.states_to_idx[elem[i - 1][0]], self.states_to_idx[elem[i][0]]] += 1
            for pair in elem:
                self.emission_matrix[self.states_to_idx[pair[0]], self.observations_to_idx[pair[1]]] += 1
        self.init_prob /= len(train_data)
        self.transition_matrix /= self.transition_matrix.sum(axis=1)[:, None]
        self.emission_matrix += 1
        self.emission_matrix /= (self.emission_matrix.sum(axis=1)[:, None])

    def __viterbi(self, obs_sequence_indices):
        """Decode incoming sequence of observations into the most propable sequence of hidden states using Viterbi algorithm
        : param obs_sequence_indices: list of observations indices
        :return: list of hidden states indices
        """
        tmp = [0] * self.states_num

        delta = [tmp[:]]  # Compute initial state probabilities
        for i in range(self.states_num):
            delta[0][i] = self.init_prob[0, i] * self.emission_matrix[i, obs_sequence_indices[0]]

        phi = [tmp[:]]

        for obs in obs_sequence_indices[1:]:  # For all observations except the inital one
            delta_t = tmp[:]
            phi_t = tmp[:]
            for j in range(self.states_num):  # Following formula 33 in Rabiner'89
                tdelta = tmp[:]
                tphimax = -1.0
                for i in range(self.states_num):
                    tphi_tmp = delta[-1][i] * self.transition_matrix[i, j]
                    if (tphi_tmp > tphimax):
                        tphimax = tphi_tmp
                        phi_t[j] = i
                    tdelta[i] = tphi_tmp * self.emission_matrix[j, obs]
                delta_t[j] = max(tdelta)
            delta.append(delta_t)
            phi.append(phi_t)

        # Backtrack the path through the states  (Formula 34 in Rabiner'89)
        #
        tmax = -1.0
        for i in range(self.states_num):
            if (delta[-1][i] > tmax):
                tmax = delta[-1][i]
                state_seq = [i]  # Last state with maximum probability

        phi.reverse()  # Because we start from the end of the sequence
        for tphi in phi[:-1]:
            state_seq.append(tphi[state_seq[-1]])
        return reversed(state_seq)

    def predict(self, obser_seq):
        """Decode observable sequences using Viterbi algorithm
        :param obser_seq: list of sentences where each sentence is represented by list of observations
        :return: list of the most probable hidden states

        Example: obser_seq = [['The','military','says','the','blast'],
                              ['The','attack','prompted','Scandinavian','monitors','overseeing','Sri','Lanka']]
        """
        pred = []
        for seq in obser_seq:
            seq_ind = [self.observations_to_idx[el] if el in self.observations
                       else self.observations_to_idx['Unk'] for el in seq]
            seq_pred = self.__viterbi(seq_ind)
            seqs = [self.inverse_idx_states[el] for el in seq_pred]
            pred.append(seqs)
        return pred
