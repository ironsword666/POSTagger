
import torch

def viterbi():
    '''

    '''
    pass

def crf(scores, tags, mask, transition):
    '''
    Params:
        criterion: loss function
        scores (Tensor(batch, seq_len, tag_nums)): ...
        tags (Tensor(batch, seq_len)): ...
        mask (Tensor(batch, seq_len)): mask <bos> <eos > and <pad>
        transition (Tensor(tag_nums, tag_nums)): transition matrix, transition_ij is score of tag_i transit to tag_j
    '''

    batch_size, seq_len, _ = scores.size()
    lens = mask.sum(dim=1) + 2

    # links[*, k, i, j] <=> e(k, j) + t(i, j) <=> k is labeled as tag_j and k-1 is labeled as tag_i
    # Tensor(batch, seq_len, 1, tag_nums) + Tensor(tag_nums, tag_nums) -> Tensor(batch, seq_len, tag_nums, tag_nums)
    links = scores.unsqueeze(dim=2) + transition

    # zs[*, k, j] is logsumexp of scores of sequence end in k, and k is labeled as tag_j
    # Tensor(batch, seq_len, tag_nums)
    alpha = scores.new_ones(*scores.size())
    # TODO how to handle <pad> <bos> and <eos>
    # here, <bos> and <eos> can emit to all tags, and although we calculate tokens exceed sentence length, we ignore the result
    alpha[:, 0] = scores[:, 0]

    for i in range(1, seq_len):
        # logsumexp((batch, tag_nums, 1) + (batch, tag_nums, tag_nums), 1) -> (batch, tag_nums)
        alpha[:, i] = torch.logsumexp(alpha[:, i-1].unsqueeze(dim=-1) + links[:, i], dim=1)

    logZ = []
    gold_scores = []
    for i in range(batch_size):
        # logZ
        logZ.append(torch.logsumexp(alpha[i, lens[i]-1], dim=0))
        # gold score
        gold_tags = tags[i]
        old_state = gold_tags[0]
        gold_score = scores[i, 0, old_state]
        for j in range(1, lens[i]):
            new_state = gold_tags[j]
            gold_score += scores[i, j, new_state] + transition[old_state, new_state]
            old_state = new_state
        gold_scores.append(gold_score)

    # Tensor(batch)
    loss = torch.stack(logZ) - torch.stack(gold_scores)

    # TODO  / mask.sum() ?
    return loss.sum()
    







