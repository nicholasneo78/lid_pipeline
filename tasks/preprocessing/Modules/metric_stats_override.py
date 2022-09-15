"""The ``metric_stats`` module provides an abstract class for storing
statistics produced over the course of an experiment and summarizing them.
Authors:
 * Peter Plantinga 2020
 * Mirco Ravanelli 2020
"""
import torch
import sys
from joblib import Parallel, delayed
import collections

EDIT_SYMBOLS = {
    "eq": "=",  # when tokens are equal
    "ins": "I",
    "del": "D",
    "sub": "S",
}

def _batch_to_dict_format(ids, seqs):
    # Used by wer_details_for_batch
    return dict(zip(ids, seqs))

def _print_alignments_global_header(
    empty_symbol="<eps>", separator=" ; ", file=sys.stdout
):
    print("=" * 80, file=file)
    print("ALIGNMENTS", file=file)
    print("", file=file)
    print("Format:", file=file)
    print("<utterance-id>, WER DETAILS", file=file)
    # Print the format with the actual
    # print_alignment function, using artificial data:
    a = ["reference", "on", "the", "first", "line"]
    b = ["and", "hypothesis", "on", "the", "third"]
    alignment = [
        (EDIT_SYMBOLS["ins"], None, 0),
        (EDIT_SYMBOLS["sub"], 0, 1),
        (EDIT_SYMBOLS["eq"], 1, 2),
        (EDIT_SYMBOLS["eq"], 2, 3),
        (EDIT_SYMBOLS["sub"], 3, 4),
        (EDIT_SYMBOLS["del"], 4, None),
    ]
    _print_alignment(
        alignment,
        a,
        b,
        file=file,
        empty_symbol=empty_symbol,
        separator=separator,
    )

def _print_alignment(
    alignment, a, b, empty_symbol="<eps>", separator=" ; ", file=sys.stdout
):
    # First, get equal length text for all:
    a_padded = []
    b_padded = []
    ops_padded = []
    for op, i, j in alignment:  # i indexes a, j indexes b
        op_string = str(op)
        a_string = str(a[i]) if i is not None else empty_symbol
        b_string = str(b[j]) if j is not None else empty_symbol
        # NOTE: the padding does not actually compute printed length,
        # but hopefully we can assume that printed length is
        # at most the str len
        pad_length = max(len(op_string), len(a_string), len(b_string))
        a_padded.append(a_string.center(pad_length))
        b_padded.append(b_string.center(pad_length))
        ops_padded.append(op_string.center(pad_length))
    # Then print, in the order Ref, op, Hyp
    print(separator.join(a_padded), file=file)
    print(separator.join(ops_padded), file=file)
    print(separator.join(b_padded), file=file)

def _print_alignment_header(wer_details, file=sys.stdout):
    print("=" * 80, file=file)
    print(
        "{key}, %WER {WER:.2f} [ {num_edits} / {num_ref_tokens}, {insertions} ins, {deletions} del, {substitutions} sub ]".format(  # noqa
            **wer_details
        ),
        file=file,
    )

def print_alignments(
    details_by_utterance,
    file=sys.stdout,
    empty_symbol="<eps>",
    separator=" ; ",
    print_header=True,
    sample_separator=None,
):
    """Print WER summary and alignments.
    Arguments
    ---------
    details_by_utterance : list
        List of wer details by utterance,
        see ``speechbrain.utils.edit_distance.wer_details_by_utterance``
        for format. Has to have alignments included.
    file : stream
        Where to write. (default: sys.stdout)
    empty_symbol : str
        Symbol to use when aligning to nothing.
    separator : str
        String that separates each token in the output. Note the spaces in the
        default.
    print_header: bool
        Whether to print headers
    sample_separator: str
        A separator to put between samples (optional)
    """
    if print_header:
        _print_alignments_global_header(
            file=file, empty_symbol=empty_symbol, separator=separator
        )
    for dets in details_by_utterance:
        if dets["scored"]:
            if print_header:
                _print_alignment_header(dets, file=file)
            _print_alignment(
                dets["alignment"],
                dets["ref_tokens"],
                dets["hyp_tokens"],
                file=file,
                empty_symbol=empty_symbol,
                separator=separator,
            )
            if sample_separator:
                print(sample_separator, file=file)

def print_wer_summary(wer_details, file=sys.stdout):
    """Prints out WER summary details in human-readable format.
    This function essentially mirrors the Kaldi compute-wer output format.
    Arguments
    ---------
    wer_details : dict
        Dict of wer summary details,
        see ``speechbrain.utils.edit_distance.wer_summary``
        for format.
    file : stream
        Where to write. (default: sys.stdout)
    """
    print(
        "%WER {WER:.2f} [ {num_edits} / {num_scored_tokens}, {insertions} ins, {deletions} del, {substitutions} sub ]".format(  # noqa
            **wer_details
        ),
        file=file,
        end="",
    )
    print(
        " [PARTIAL]"
        if wer_details["num_scored_sents"] < wer_details["num_ref_sents"]
        else "",
        file=file,
    )
    print(
        "%SER {SER:.2f} [ {num_erraneous_sents} / {num_scored_sents} ]".format(
            **wer_details
        ),
        file=file,
    )
    print(
        "Scored {num_scored_sents} sentences, {num_absent_sents} not present in hyp.".format(  # noqa
            **wer_details
        ),
        file=file,
    )

def split_word(sequences, space="_"):
    """Split word sequences into character sequences.
    Arguments
    ---------
    sequences : list
        Each item contains a list, and this list contains a words sequence.
    space : string
        The token represents space. Default: _
    Returns
    -------
    The list contains word sequences for each sentence.
    Example
    -------
    >>> sequences = [['ab', 'c', 'de'], ['efg', 'hi']]
    >>> results = split_word(sequences)
    >>> results
    [['a', 'b', '_', 'c', '_', 'd', 'e'], ['e', 'f', 'g', '_', 'h', 'i']]
    """
    results = []
    for seq in sequences:
        chars = list(space.join(seq))
        results.append(chars)
    return 

def merge_char(sequences, space="_"):
    """Merge characters sequences into word sequences.
    Arguments
    ---------
    sequences : list
        Each item contains a list, and this list contains a character sequence.
    space : string
        The token represents space. Default: _
    Returns
    -------
    The list contains word sequences for each sentence.
    Example
    -------
    >>> sequences = [["a", "b", "_", "c", "_", "d", "e"], ["e", "f", "g", "_", "h", "i"]]
    >>> results = merge_char(sequences)
    >>> results
    [['ab', 'c', 'de'], ['efg', 'hi']]
    """
    results = []
    for seq in sequences:
        words = "".join(seq).split(space)
        results.append(words)
    return 

def undo_padding(batch, lengths):
    """Produces Python lists given a batch of sentences with
    their corresponding relative lengths.
    Arguments
    ---------
    batch : tensor
        Batch of sentences gathered in a batch.
    lengths : tensor
        Relative length of each sentence in the batch.
    Example
    -------
    >>> batch=torch.rand([4,100])
    >>> lengths=torch.tensor([0.5,0.6,0.7,1.0])
    >>> snt_list=undo_padding(batch, lengths)
    >>> len(snt_list)
    4
    """
    batch_max_len = batch.shape[1]
    as_list = []
    for seq, seq_length in zip(batch, lengths):
        actual_size = int(torch.round(seq_length * batch_max_len))
        seq_true = seq.narrow(0, 0, actual_size)
        as_list.append(seq_true.tolist())
    return 

def wer_details_for_batch(ids, refs, hyps, compute_alignments=False):
    """Convenient batch interface for ``wer_details_by_utterance``.
    ``wer_details_by_utterance`` can handle missing hypotheses, but
    sometimes (e.g. CTC training with greedy decoding) they are not needed,
    and this is a convenient interface in that case.
    Arguments
    ---------
    ids : list, torch.tensor
        Utterance ids for the batch.
    refs : list, torch.tensor
        Reference sequences.
    hyps : list, torch.tensor
        Hypothesis sequences.
    compute_alignments : bool, optional
        Whether to compute alignments or not. If computed, the details
        will also store the refs and hyps. (default: False)
    Returns
    -------
    list
        See ``wer_details_by_utterance``
    Example
    -------
    >>> ids = [['utt1'], ['utt2']]
    >>> refs = [[['a','b','c']], [['d','e']]]
    >>> hyps = [[['a','b','d']], [['d','e']]]
    >>> wer_details = []
    >>> for ids_batch, refs_batch, hyps_batch in zip(ids, refs, hyps):
    ...     details = wer_details_for_batch(ids_batch, refs_batch, hyps_batch)
    ...     wer_details.extend(details)
    >>> print(wer_details[0]['key'], ":",
    ...     "{:.2f}".format(wer_details[0]['WER']))
    utt1 : 33.33
    """
    refs = _batch_to_dict_format(ids, refs)
    hyps = _batch_to_dict_format(ids, hyps)
    return wer_details_by_utterance(
        refs, hyps, compute_alignments=compute_alignments, scoring_mode="strict"
    )

def wer_details_by_utterance(
    ref_dict, hyp_dict, compute_alignments=False, scoring_mode="strict"
):
    """Computes a wealth WER info about each single utterance.
    This info can then be used to compute summary details (WER, SER).
    Arguments
    ---------
    ref_dict : dict
        Should be indexable by utterance ids, and return the reference tokens
        for each utterance id as iterable
    hyp_dict : dict
        Should be indexable by utterance ids, and return
        the hypothesis tokens for each utterance id as iterable
    compute_alignments : bool
        Whether alignments should also be saved.
        This also saves the tokens themselves, as they are probably
        required for printing the alignments.
    scoring_mode : {'strict', 'all', 'present'}
        How to deal with missing hypotheses (reference utterance id
        not found in hyp_dict).
        * 'strict': Raise error for missing hypotheses.
        * 'all': Score missing hypotheses as empty.
        * 'present': Only score existing hypotheses.
    Returns
    -------
    list
        A list with one entry for every reference utterance. Each entry is a
        dict with keys:
        * "key": utterance id
        * "scored": (bool) Whether utterance was scored.
        * "hyp_absent": (bool) True if a hypothesis was NOT found.
        * "hyp_empty": (bool) True if hypothesis was considered empty
          (either because it was empty, or not found and mode 'all').
        * "num_edits": (int) Number of edits in total.
        * "num_ref_tokens": (int) Number of tokens in the reference.
        * "WER": (float) Word error rate of the utterance.
        * "insertions": (int) Number of insertions.
        * "deletions": (int) Number of deletions.
        * "substitutions": (int) Number of substitutions.
        * "alignment": If compute_alignments is True, alignment as list,
          see ``speechbrain.utils.edit_distance.alignment``.
          If compute_alignments is False, this is None.
        * "ref_tokens": (iterable) The reference tokens
          only saved if alignments were computed, else None.
        * "hyp_tokens": (iterable) the hypothesis tokens,
          only saved if alignments were computed, else None.
    Raises
    ------
    KeyError
        If scoring mode is 'strict' and a hypothesis is not found.
    """
    details_by_utterance = []
    for key, ref_tokens in ref_dict.items():
        # Initialize utterance_details
        utterance_details = {
            "key": key,
            "scored": False,
            "hyp_absent": None,
            "hyp_empty": None,
            "num_edits": None,
            "num_ref_tokens": len(ref_tokens),
            "WER": None,
            "insertions": None,
            "deletions": None,
            "substitutions": None,
            "alignment": None,
            "ref_tokens": ref_tokens if compute_alignments else None,
            "hyp_tokens": None,
        }
        if key in hyp_dict:
            utterance_details.update({"hyp_absent": False})
            hyp_tokens = hyp_dict[key]
        elif scoring_mode == "all":
            utterance_details.update({"hyp_absent": True})
            hyp_tokens = []
        elif scoring_mode == "present":
            utterance_details.update({"hyp_absent": True})
            details_by_utterance.append(utterance_details)
            continue  # Skip scoring this utterance
        elif scoring_mode == "strict":
            raise KeyError(
                "Key "
                + key
                + " in reference but missing in hypothesis and strict mode on."
            )
        else:
            raise ValueError("Invalid scoring mode: " + scoring_mode)
        # Compute edits for this utterance
        table = op_table(ref_tokens, hyp_tokens)
        ops = count_ops(table)
        # Update the utterance-level details if we got this far:
        utterance_details.update(
            {
                "scored": True,
                "hyp_empty": True
                if len(hyp_tokens) == 0
                else False,  # This also works for e.g. torch tensors
                "num_edits": sum(ops.values()),
                "num_ref_tokens": len(ref_tokens),
                "WER": 100.0 * sum(ops.values()) / len(ref_tokens),
                "insertions": ops["insertions"],
                "deletions": ops["deletions"],
                "substitutions": ops["substitutions"],
                "alignment": alignment(table) if compute_alignments else None,
                "ref_tokens": ref_tokens if compute_alignments else None,
                "hyp_tokens": hyp_tokens if compute_alignments else None,
            }
        )
        details_by_utterance.append(utterance_details)
    return details_by_utterance

def op_table(a, b):
    """Table of edit operations between a and b.
    Solves for the table of edit operations, which is mainly used to
    compute word error rate. The table is of size ``[|a|+1, |b|+1]``,
    and each point ``(i, j)`` in the table has an edit operation. The
    edit operations can be deterministically followed backwards to
    find the shortest edit path to from ``a[:i-1] to b[:j-1]``. Indexes
    of zero (``i=0`` or ``j=0``) correspond to an empty sequence.
    The algorithm itself is well known, see
    `Levenshtein distance <https://en.wikipedia.org/wiki/Levenshtein_distance>`_
    Note that in some cases there are multiple valid edit operation
    paths which lead to the same edit distance minimum.
    Arguments
    ---------
    a : iterable
        Sequence for which the edit operations are solved.
    b : iterable
        Sequence for which the edit operations are solved.
    Returns
    -------
    list
        List of lists, Matrix, Table of edit operations.
    Example
    -------
    >>> ref = [1,2,3]
    >>> hyp = [1,2,4]
    >>> for row in op_table(ref, hyp):
    ...     print(row)
    ['=', 'I', 'I', 'I']
    ['D', '=', 'I', 'I']
    ['D', 'D', '=', 'I']
    ['D', 'D', 'D', 'S']
    """
    # For the dynamic programming algorithm, only two rows are really needed:
    # the one currently being filled in, and the previous one
    # The following is also the right initialization
    prev_row = [j for j in range(len(b) + 1)]
    curr_row = [0] * (len(b) + 1)  # Just init to zero
    # For the edit operation table we will need the whole matrix.
    # We will initialize the table with no-ops, so that we only need to change
    # where an edit is made.
    table = [
        [EDIT_SYMBOLS["eq"] for j in range(len(b) + 1)]
        for i in range(len(a) + 1)
    ]
    # We already know the operations on the first row and column:
    for i in range(len(a) + 1):
        table[i][0] = EDIT_SYMBOLS["del"]
    for j in range(len(b) + 1):
        table[0][j] = EDIT_SYMBOLS["ins"]
    table[0][0] = EDIT_SYMBOLS["eq"]
    # The rest of the table is filled in row-wise:
    for i, a_token in enumerate(a, start=1):
        curr_row[0] += 1  # This trick just deals with the first column.
        for j, b_token in enumerate(b, start=1):
            # The dynamic programming algorithm cost rules
            insertion_cost = curr_row[j - 1] + 1
            deletion_cost = prev_row[j] + 1
            substitution = 0 if a_token == b_token else 1
            substitution_cost = prev_row[j - 1] + substitution
            # Here copying the Kaldi compute-wer comparison order, which in
            # ties prefers:
            # insertion > deletion > substitution
            if (
                substitution_cost < insertion_cost
                and substitution_cost < deletion_cost
            ):
                curr_row[j] = substitution_cost
                # Again, note that if not substitution, the edit table already
                # has the correct no-op symbol.
                if substitution:
                    table[i][j] = EDIT_SYMBOLS["sub"]
            elif deletion_cost < insertion_cost:
                curr_row[j] = deletion_cost
                table[i][j] = EDIT_SYMBOLS["del"]
            else:
                curr_row[j] = insertion_cost
                table[i][j] = EDIT_SYMBOLS["ins"]
        # Move to the next row:
        prev_row[:] = curr_row[:]
    return table

def count_ops(table):
    """Count the edit operations in the shortest edit path in edit op table.
    Walks back an edit operations table produced by table(a, b) and
    counts the number of insertions, deletions, and substitutions in the
    shortest edit path. This information is typically used in speech
    recognition to report the number of different error types separately.
    Arguments
    ----------
    table : list
        Edit operations table from ``op_table(a, b)``.
    Returns
    -------
    collections.Counter
        The counts of the edit operations, with keys:
        * "insertions"
        * "deletions"
        * "substitutions"
        NOTE: not all of the keys might appear explicitly in the output,
        but for the missing keys collections. The counter will return 0.
    Example
    -------
    >>> table = [['I', 'I', 'I', 'I'],
    ...          ['D', '=', 'I', 'I'],
    ...          ['D', 'D', '=', 'I'],
    ...          ['D', 'D', 'D', 'S']]
    >>> print(count_ops(table))
    Counter({'substitutions': 1})
    """
    edits = collections.Counter()
    # Walk back the table, gather the ops.
    i = len(table) - 1
    j = len(table[0]) - 1
    while not (i == 0 and j == 0):
        if i == 0:
            edits["insertions"] += 1
            j -= 1
        elif j == 0:
            edits["deletions"] += 1
            i -= 1
        else:
            if table[i][j] == EDIT_SYMBOLS["ins"]:
                edits["insertions"] += 1
                j -= 1
            elif table[i][j] == EDIT_SYMBOLS["del"]:
                edits["deletions"] += 1
                i -= 1
            else:
                if table[i][j] == EDIT_SYMBOLS["sub"]:
                    edits["substitutions"] += 1
                i -= 1
                j -= 1
    return edits

def alignment(table):
    """Get the edit distance alignment from an edit op table.
    Walks back an edit operations table, produced by calling ``table(a, b)``,
    and collects the edit distance alignment of a to b. The alignment
    shows which token in a corresponds to which token in b. Note that the
    alignment is monotonic, one-to-zero-or-one.
    Arguments
    ----------
    table : list
        Edit operations table from ``op_table(a, b)``.
    Returns
    -------
    list
        Schema: ``[(str <edit-op>, int-or-None <i>, int-or-None <j>),]``
        List of edit operations, and the corresponding indices to a and b.
        See the EDIT_SYMBOLS dict for the edit-ops.
        The i indexes a, j indexes b, and the indices can be None, which means
        aligning to nothing.
    Example
    -------
    >>> # table for a=[1,2,3], b=[1,2,4]:
    >>> table = [['I', 'I', 'I', 'I'],
    ...          ['D', '=', 'I', 'I'],
    ...          ['D', 'D', '=', 'I'],
    ...          ['D', 'D', 'D', 'S']]
    >>> print(alignment(table))
    [('=', 0, 0), ('=', 1, 1), ('S', 2, 2)]
    """
    # The alignment will be the size of the longer sequence.
    # form: [(op, a_index, b_index)], index is None when aligned to empty
    alignment = []
    # Now we'll walk back the table to get the alignment.
    i = len(table) - 1
    j = len(table[0]) - 1
    while not (i == 0 and j == 0):
        if i == 0:
            j -= 1
            alignment.insert(0, (EDIT_SYMBOLS["ins"], None, j))
        elif j == 0:
            i -= 1
            alignment.insert(0, (EDIT_SYMBOLS["del"], i, None))
        else:
            if table[i][j] == EDIT_SYMBOLS["ins"]:
                j -= 1
                alignment.insert(0, (EDIT_SYMBOLS["ins"], None, j))
            elif table[i][j] == EDIT_SYMBOLS["del"]:
                i -= 1
                alignment.insert(0, (EDIT_SYMBOLS["del"], i, None))
            elif table[i][j] == EDIT_SYMBOLS["sub"]:
                i -= 1
                j -= 1
                alignment.insert(0, (EDIT_SYMBOLS["sub"], i, j))
            else:
                i -= 1
                j -= 1
                alignment.insert(0, (EDIT_SYMBOLS["eq"], i, j))
    return 

def wer_summary(details_by_utterance):
    """
    Computes summary stats from the output of details_by_utterance
    Summary stats like WER
    Arguments
    ---------
    details_by_utterance : list
        See the output of wer_details_by_utterance
    Returns
    -------
    dict
        Dictionary with keys:
        * "WER": (float) Word Error Rate.
        * "SER": (float) Sentence Error Rate (percentage of utterances
          which had at least one error).
        * "num_edits": (int) Total number of edits.
        * "num_scored_tokens": (int) Total number of tokens in scored
          reference utterances (a missing hypothesis might still
          have been scored with 'all' scoring mode).
        * "num_erraneous_sents": (int) Total number of utterances
          which had at least one error.
        * "num_scored_sents": (int) Total number of utterances
          which were scored.
        * "num_absent_sents": (int) Hypotheses which were not found.
        * "num_ref_sents": (int) Number of all reference utterances.
        * "insertions": (int) Total number of insertions.
        * "deletions": (int) Total number of deletions.
        * "substitutions": (int) Total number of substitutions.
        NOTE: Some cases lead to ambiguity over number of
        insertions, deletions and substitutions. We
        aim to replicate Kaldi compute_wer numbers.
    """
    # Build the summary details:
    ins = dels = subs = 0
    num_scored_tokens = (
        num_scored_sents
    ) = num_edits = num_erraneous_sents = num_absent_sents = num_ref_sents = 0
    for dets in details_by_utterance:
        num_ref_sents += 1
        if dets["scored"]:
            num_scored_sents += 1
            num_scored_tokens += dets["num_ref_tokens"]
            ins += dets["insertions"]
            dels += dets["deletions"]
            subs += dets["substitutions"]
            num_edits += dets["num_edits"]
            if dets["num_edits"] > 0:
                num_erraneous_sents += 1
        if dets["hyp_absent"]:
            num_absent_sents += 1
    wer_details = {
        "WER": 100.0 * num_edits / num_scored_tokens,
        "SER": 100.0 * num_erraneous_sents / num_scored_sents,
        "num_edits": num_edits,
        "num_scored_tokens": num_scored_tokens,
        "num_erraneous_sents": num_erraneous_sents,
        "num_scored_sents": num_scored_sents,
        "num_absent_sents": num_absent_sents,
        "num_ref_sents": num_ref_sents,
        "insertions": ins,
        "deletions": dels,
        "substitutions": subs,
    }
    return 

class MetricStats:
    """A default class for storing and summarizing arbitrary metrics.
    More complex metrics can be created by sub-classing this class.
    Arguments
    ---------
    metric : function
        The function to use to compute the relevant metric. Should take
        at least two arguments (predictions and targets) and can
        optionally take the relative lengths of either or both arguments.
        Not usually used in sub-classes.
    batch_eval: bool
        When True it feeds the evaluation metric with the batched input.
        When False and n_jobs=1, it performs metric evaluation one-by-one
        in a sequential way. When False and n_jobs>1, the evaluation
        runs in parallel over the different inputs using joblib.
    n_jobs : int
        The number of jobs to use for computing the metric. If this is
        more than one, every sample is processed individually, otherwise
        the whole batch is passed at once.
    Example
    -------
    >>> from speechbrain.nnet.losses import l1_loss
    >>> loss_stats = MetricStats(metric=l1_loss)
    >>> loss_stats.append(
    ...      ids=["utterance1", "utterance2"],
    ...      predictions=torch.tensor([[0.1, 0.2], [0.2, 0.3]]),
    ...      targets=torch.tensor([[0.1, 0.2], [0.1, 0.2]]),
    ...      reduction="batch",
    ... )
    >>> stats = loss_stats.summarize()
    >>> stats['average']
    0.050...
    >>> stats['max_score']
    0.100...
    >>> stats['max_id']
    'utterance2'
    """

    def __init__(self, metric, n_jobs=1, batch_eval=True):
        self.metric = metric
        self.n_jobs = n_jobs
        self.batch_eval = batch_eval
        self.clear()

    def clear(self):
        """Creates empty container for storage, removing existing stats."""
        self.scores = []
        self.ids = []
        self.summary = {}

    def append(self, ids, *args, **kwargs):
        """Store a particular set of metric scores.
        Arguments
        ---------
        ids : list
            List of ids corresponding to utterances.
        *args, **kwargs
            Arguments to pass to the metric function.
        """
        self.ids.extend(ids)

        # Batch evaluation
        if self.batch_eval:
            scores = self.metric(*args, **kwargs).detach()

        else:
            if "predict" not in kwargs or "target" not in kwargs:
                raise ValueError(
                    "Must pass 'predict' and 'target' as kwargs if batch_eval=False"
                )
            if self.n_jobs == 1:
                # Sequence evaluation (loop over inputs)
                scores = sequence_evaluation(metric=self.metric, **kwargs)
            else:
                # Multiprocess evaluation
                scores = multiprocess_evaluation(
                    metric=self.metric, n_jobs=self.n_jobs, **kwargs
                )

        self.scores.extend(scores)

    def summarize(self, field=None):
        """Summarize the metric scores, returning relevant stats.
        Arguments
        ---------
        field : str
            If provided, only returns selected statistic. If not,
            returns all computed statistics.
        Returns
        -------
        float or dict
            Returns a float if ``field`` is provided, otherwise
            returns a dictionary containing all computed stats.
        """
        min_index = torch.argmin(torch.tensor(self.scores))
        max_index = torch.argmax(torch.tensor(self.scores))
        self.summary = {
            "average": float(sum(self.scores) / len(self.scores)),
            "min_score": float(self.scores[min_index]),
            "min_id": self.ids[min_index],
            "max_score": float(self.scores[max_index]),
            "max_id": self.ids[max_index],
        }

        if field is not None:
            return self.summary[field]
        else:
            return self.summary

    def write_stats(self, filestream, verbose=False):
        """Write all relevant statistics to file.
        Arguments
        ---------
        filestream : file-like object
            A stream for the stats to be written to.
        verbose : bool
            Whether to also print the stats to stdout.
        """
        if not self.summary:
            self.summarize()

        message = f"Average score: {self.summary['average']}\n"
        message += f"Min error: {self.summary['min_score']} "
        message += f"id: {self.summary['min_id']}\n"
        message += f"Max error: {self.summary['max_score']} "
        message += f"id: {self.summary['max_id']}\n"

        filestream.write(message)
        if verbose:
            print(message)


def multiprocess_evaluation(metric, predict, target, lengths=None, n_jobs=8):
    """Runs metric evaluation if parallel over multiple jobs."""
    if lengths is not None:
        lengths = (lengths * predict.size(1)).round().int().cpu()
        predict = [p[:length].cpu() for p, length in zip(predict, lengths)]
        target = [t[:length].cpu() for t, length in zip(target, lengths)]

    while True:
        try:
            scores = Parallel(n_jobs=n_jobs, timeout=30)(
                delayed(metric)(p, t) for p, t in zip(predict, target)
            )
            break
        except Exception as e:
            print(e)
            print("Evaluation timeout...... (will try again)")

    return scores


def sequence_evaluation(metric, predict, target, lengths=None):
    """Runs metric evaluation sequentially over the inputs."""
    if lengths is not None:
        lengths = (lengths * predict.size(1)).round().int().cpu()
        predict = [p[:length].cpu() for p, length in zip(predict, lengths)]
        target = [t[:length].cpu() for t, length in zip(target, lengths)]

    scores = []
    for p, t in zip(predict, target):
        score = metric(p, t)
        scores.append(score)
    return scores


class ErrorRateStats(MetricStats):
    """A class for tracking error rates (e.g., WER, PER).
    Arguments
    ---------
    merge_tokens : bool
        Whether to merge the successive tokens (used for e.g.,
        creating words out of character tokens).
        See ``speechbrain.dataio.dataio.merge_char``.
    split_tokens : bool
        Whether to split tokens (used for e.g. creating
        characters out of word tokens).
        See ``speechbrain.dataio.dataio.split_word``.
    space_token : str
        The character to use for boundaries. Used with ``merge_tokens``
        this represents character to split on after merge.
        Used with ``split_tokens`` the sequence is joined with
        this token in between, and then the whole sequence is split.
    Example
    -------
    >>> cer_stats = ErrorRateStats()
    >>> i2l = {0: 'a', 1: 'b'}
    >>> cer_stats.append(
    ...     ids=['utterance1'],
    ...     predict=torch.tensor([[0, 1, 1]]),
    ...     target=torch.tensor([[0, 1, 0]]),
    ...     target_len=torch.ones(1),
    ...     ind2lab=lambda batch: [[i2l[int(x)] for x in seq] for seq in batch],
    ... )
    >>> stats = cer_stats.summarize()
    >>> stats['WER']
    33.33...
    >>> stats['insertions']
    0
    >>> stats['deletions']
    0
    >>> stats['substitutions']
    1
    """

    def __init__(self, merge_tokens=False, split_tokens=False, space_token="_"):
        self.clear()
        self.merge_tokens = merge_tokens
        self.split_tokens = split_tokens
        self.space_token = space_token

    def append(
        self,
        ids,
        predict,
        target,
        predict_len=None,
        target_len=None,
        ind2lab=None,
    ):
        """Add stats to the relevant containers.
        * See MetricStats.append()
        Arguments
        ---------
        ids : list
            List of ids corresponding to utterances.
        predict : torch.tensor
            A predicted output, for comparison with the target output
        target : torch.tensor
            The correct reference output, for comparison with the prediction.
        predict_len : torch.tensor
            The predictions relative lengths, used to undo padding if
            there is padding present in the predictions.
        target_len : torch.tensor
            The target outputs' relative lengths, used to undo padding if
            there is padding present in the target.
        ind2lab : callable
            Callable that maps from indices to labels, operating on batches,
            for writing alignments.
        """
        self.ids.extend(ids)

        if predict_len is not None:
            predict = undo_padding(predict, predict_len)

        if target_len is not None:
            target = undo_padding(target, target_len)

        if ind2lab is not None:
            predict = ind2lab(predict)
            target = ind2lab(target)

        if self.merge_tokens:
            predict = merge_char(predict, space=self.space_token)
            target = merge_char(target, space=self.space_token)

        if self.split_tokens:
            predict = split_word(predict, space=self.space_token)
            target = split_word(target, space=self.space_token)

        scores = wer_details_for_batch(ids, target, predict, True)

        self.scores.extend(scores)

    def summarize(self, field=None):
        """Summarize the error_rate and return relevant statistics.
        * See MetricStats.summarize()
        """
        self.summary = wer_summary(self.scores)

        # Add additional, more generic key
        self.summary["error_rate"] = self.summary["WER"]

        if field is not None:
            return self.summary[field]
        else:
            return self.summary

    def write_stats(self, filestream):
        """Write all relevant info (e.g., error rate alignments) to file.
        * See MetricStats.write_stats()
        """
        if not self.summary:
            self.summarize()

        print_wer_summary(self.summary, filestream)
        print_alignments(self.scores, filestream)


class BinaryMetricStats(MetricStats):
    """Tracks binary metrics, such as precision, recall, F1, EER, etc.
    """

    def __init__(self, positive_label=1):
        self.clear()
        self.positive_label = positive_label

    def clear(self):
        """Clears the stored metrics."""
        self.ids = []
        self.scores = []
        self.labels = []
        self.summary = {}

    def append(self, ids, scores, labels):
        """Appends scores and labels to internal lists.
        Does not compute metrics until time of summary, since
        automatic thresholds (e.g., EER) need full set of scores.
        Arguments
        ---------
        ids : list
            The string ids for the samples
        """
        self.ids.extend(ids)
        self.scores.extend(scores.detach())
        self.labels.extend(labels.detach())

    def summarize(
        self, field=None, threshold=None, max_samples=None, beta=1, eps=1e-8
    ):
        """Compute statistics using a full set of scores.
        Full set of fields:
         - TP - True Positive
         - TN - True Negative
         - FP - False Positive
         - FN - False Negative
         - FAR - False Acceptance Rate
         - FRR - False Rejection Rate
         - DER - Detection Error Rate (EER if no threshold passed)
         - threshold - threshold (EER threshold if no threshold passed)
         - precision - Precision (positive predictive value)
         - recall - Recall (sensitivity)
         - F-score - Balance of precision and recall (equal if beta=1)
         - MCC - Matthews Correlation Coefficient
        Arguments
        ---------
        field : str
            A key for selecting a single statistic. If not provided,
            a dict with all statistics is returned.
        threshold : float
            If no threshold is provided, equal error rate is used.
        max_samples: float
            How many samples to keep for postive/negative scores.
            If no max_samples is provided, all scores are kept.
            Only effective when threshold is None.
        beta : float
            How much to weight precision vs recall in F-score. Default
            of 1. is equal weight, while higher values weight recall
            higher, and lower values weight precision higher.
        eps : float
            A small value to avoid dividing by zero.
        """

        if isinstance(self.scores, list):
            self.scores = torch.stack(self.scores)
            self.labels = torch.stack(self.labels)

        if threshold is None:
            positive_scores = self.scores[
                (self.labels == 1).nonzero(as_tuple=True)
            ]
            negative_scores = self.scores[
                (self.labels == 0).nonzero(as_tuple=True)
            ]
            if max_samples is not None:
                if len(positive_scores) > max_samples:
                    positive_scores, _ = torch.sort(positive_scores)
                    positive_scores = positive_scores[
                        [
                            i
                            for i in range(
                                0,
                                len(positive_scores),
                                int(len(positive_scores) / max_samples),
                            )
                        ]
                    ]
                if len(negative_scores) > max_samples:
                    negative_scores, _ = torch.sort(negative_scores)
                    negative_scores = negative_scores[
                        [
                            i
                            for i in range(
                                0,
                                len(negative_scores),
                                int(len(negative_scores) / max_samples),
                            )
                        ]
                    ]

            eer, threshold = EER(positive_scores, negative_scores)

        pred = (self.scores > threshold).float()
        true = self.labels

        TP = self.summary["TP"] = float(pred.mul(true).sum())
        TN = self.summary["TN"] = float((1.0 - pred).mul(1.0 - true).sum())
        FP = self.summary["FP"] = float(pred.mul(1.0 - true).sum())
        FN = self.summary["FN"] = float((1.0 - pred).mul(true).sum())

        self.summary["FAR"] = FP / (FP + TN + eps)
        self.summary["FRR"] = FN / (TP + FN + eps)
        self.summary["DER"] = (FP + FN) / (TP + TN + eps)
        self.summary["threshold"] = threshold

        self.summary["precision"] = TP / (TP + FP + eps)
        self.summary["recall"] = TP / (TP + FN + eps)
        self.summary["F-score"] = (
            (1.0 + beta ** 2.0)
            * TP
            / ((1.0 + beta ** 2.0) * TP + beta ** 2.0 * FN + FP)
        )

        self.summary["MCC"] = (TP * TN - FP * FN) / (
            (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + eps
        ) ** 0.5

        if field is not None:
            return self.summary[field], self.scores, self.labels
        else:
            return self.summary, self.scores, self.labels


def EER(positive_scores, negative_scores):
    """Computes the EER (and its threshold).
    Arguments
    ---------
    positive_scores : torch.tensor
        The scores from entries of the same class.
    negative_scores : torch.tensor
        The scores from entries of different classes.
    Example
    -------
    >>> positive_scores = torch.tensor([0.6, 0.7, 0.8, 0.5])
    >>> negative_scores = torch.tensor([0.4, 0.3, 0.2, 0.1])
    >>> val_eer, threshold = EER(positive_scores, negative_scores)
    >>> val_eer
    0.0
    """

    # Computing candidate thresholds
    thresholds, _ = torch.sort(torch.cat([positive_scores, negative_scores]))
    thresholds = torch.unique(thresholds)

    # Adding intermediate thresholds
    interm_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds, _ = torch.sort(torch.cat([thresholds, interm_thresholds]))

    # Variable to store the min FRR, min FAR and their corresponding index
    min_index = 0
    final_FRR = 0
    final_FAR = 0

    for i, cur_thresh in enumerate(thresholds):
        pos_scores_threshold = positive_scores <= cur_thresh
        FRR = (pos_scores_threshold.sum(0)).float() / positive_scores.shape[0]
        del pos_scores_threshold

        neg_scores_threshold = negative_scores > cur_thresh
        FAR = (neg_scores_threshold.sum(0)).float() / negative_scores.shape[0]
        del neg_scores_threshold

        # Finding the threshold for EER
        if (FAR - FRR).abs().item() < abs(final_FAR - final_FRR) or i == 0:
            min_index = i
            final_FRR = FRR.item()
            final_FAR = FAR.item()

    # It is possible that eer != fpr != fnr. We return (FAR  + FRR) / 2 as EER.
    EER = (final_FAR + final_FRR) / 2

    return float(EER), float(thresholds[min_index])


def minDCF(
    positive_scores, negative_scores, c_miss=1.0, c_fa=1.0, p_target=0.01
):
    """Computes the minDCF metric normally used to evaluate speaker verification
    systems. The min_DCF is the minimum of the following C_det function computed
    within the defined threshold range:
    C_det =  c_miss * p_miss * p_target + c_fa * p_fa * (1 -p_target)
    where p_miss is the missing probability and p_fa is the probability of having
    a false alarm.
    Arguments
    ---------
    positive_scores : torch.tensor
        The scores from entries of the same class.
    negative_scores : torch.tensor
        The scores from entries of different classes.
    c_miss : float
         Cost assigned to a missing error (default 1.0).
    c_fa : float
        Cost assigned to a false alarm (default 1.0).
    p_target: float
        Prior probability of having a target (default 0.01).
    Example
    -------
    >>> positive_scores = torch.tensor([0.6, 0.7, 0.8, 0.5])
    >>> negative_scores = torch.tensor([0.4, 0.3, 0.2, 0.1])
    >>> val_minDCF, threshold = minDCF(positive_scores, negative_scores)
    >>> val_minDCF
    0.0
    """

    # Computing candidate thresholds
    thresholds, _ = torch.sort(torch.cat([positive_scores, negative_scores]))
    thresholds = torch.unique(thresholds)

    # Adding intermediate thresholds
    interm_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds, _ = torch.sort(torch.cat([thresholds, interm_thresholds]))

    # Computing False Rejection Rate (miss detection)
    positive_scores = torch.cat(
        len(thresholds) * [positive_scores.unsqueeze(0)]
    )
    pos_scores_threshold = positive_scores.transpose(0, 1) <= thresholds
    p_miss = (pos_scores_threshold.sum(0)).float() / positive_scores.shape[1]
    del positive_scores
    del pos_scores_threshold

    # Computing False Acceptance Rate (false alarm)
    negative_scores = torch.cat(
        len(thresholds) * [negative_scores.unsqueeze(0)]
    )
    neg_scores_threshold = negative_scores.transpose(0, 1) > thresholds
    p_fa = (neg_scores_threshold.sum(0)).float() / negative_scores.shape[1]
    del negative_scores
    del neg_scores_threshold

    c_det = c_miss * p_miss * p_target + c_fa * p_fa * (1 - p_target)
    c_min, min_index = torch.min(c_det, dim=0)

    return float(c_min), float(thresholds[min_index])


class ClassificationStats(MetricStats):
    """Computes statistics pertaining to multi-label
    classification tasks, as well as tasks that can be loosely interpreted as such for the purpose of
    evaluations
    Example
    -------
    >>> import sys
    >>> from speechbrain.utils.metric_stats import ClassificationStats
    >>> cs = ClassificationStats()
    >>> cs.append(
    ...     ids=["ITEM1", "ITEM2", "ITEM3", "ITEM4"],
    ...     predictions=[
    ...         "M EY K AH",
    ...         "T EY K",
    ...         "B AE D",
    ...         "M EY K",
    ...     ],
    ...     targets=[
    ...         "M EY K",
    ...         "T EY K",
    ...         "B AE D",
    ...         "M EY K",
    ...     ],
    ...     categories=[
    ...         "make",
    ...         "take",
    ...         "bad",
    ...         "make"
    ...     ]
    ... )
    >>> cs.write_stats(sys.stdout)
    Overall Accuracy: 75%
    <BLANKLINE>
    Class-Wise Accuracy
    -------------------
    bad -> B AE D : 1 / 1 (100.00%)
    make -> M EY K: 1 / 2 (50.00%)
    take -> T EY K: 1 / 1 (100.00%)
    <BLANKLINE>
    Confusion
    ---------
    Target: bad -> B AE D
      -> B AE D   : 1 / 1 (100.00%)
    Target: make -> M EY K
      -> M EY K   : 1 / 2 (50.00%)
      -> M EY K AH: 1 / 2 (50.00%)
    Target: take -> T EY K
      -> T EY K   : 1 / 1 (100.00%)
    >>> summary = cs.summarize()
    >>> summary['accuracy']
    0.75
    >>> summary['classwise_stats'][('bad', 'B AE D')]
    {'total': 1.0, 'correct': 1.0, 'accuracy': 1.0}
    >>> summary['classwise_stats'][('make', 'M EY K')]
    {'total': 2.0, 'correct': 1.0, 'accuracy': 0.5}
    >>> summary['keys']
    [('bad', 'B AE D'), ('make', 'M EY K'), ('take', 'T EY K')]
    >>> summary['predictions']
    ['B AE D', 'M EY K', 'M EY K AH', 'T EY K']
    >>> summary['classwise_total']
    {('bad', 'B AE D'): 1.0, ('make', 'M EY K'): 2.0, ('take', 'T EY K'): 1.0}
    >>> summary['classwise_correct']
    {('bad', 'B AE D'): 1.0, ('make', 'M EY K'): 1.0, ('take', 'T EY K'): 1.0}
    >>> summary['classwise_accuracy']
    {('bad', 'B AE D'): 1.0, ('make', 'M EY K'): 0.5, ('take', 'T EY K'): 1.0}
    """

    def __init__(self):
        super()
        self.clear()
        self.summary = None

    def append(self, ids, predictions, targets, categories=None):
        """
        Appends inputs, predictions and targets to internal
        lists
        Arguments
        ---------
        ids: list
            the string IDs for the samples
        predictions: list
            the model's predictions (human-interpretable,
            preferably strings)
        targets: list
            the ground truths (human-interpretable, preferably strings)
        categories: list
            an additional way to classify training
            samples. If available, the categories will
            be combined with targets
        """
        self.ids.extend(ids)
        self.predictions.extend(predictions)
        self.targets.extend(targets)
        if categories is not None:
            self.categories.extend(categories)

    def summarize(self, field=None):
        """Summarize the classification metric scores
        The following statistics are computed:
        accuracy: the overall accuracy (# correct / # total)
        confusion_matrix: a dictionary of type
            {(target, prediction): num_entries} representing
            the confusion matrix
        classwise_stats: computes the total number of samples,
            the number of correct classifications and accuracy
            for each class
        keys: all available class keys, which can be either target classes
            or (category, target) tuples
        predictions: all available predictions all predicions the model
            has made
        Arguments
        ---------
        field : str
            If provided, only returns selected statistic. If not,
            returns all computed statistics.
        Returns
        -------
        float or dict
            Returns a float if ``field`` is provided, otherwise
            returns a dictionary containing all computed stats.
        """

        self._build_lookups()
        confusion_matrix = self._compute_confusion_matrix()
        self.summary = {
            "accuracy": self._compute_accuracy(),
            "confusion_matrix": confusion_matrix,
            "classwise_stats": self._compute_classwise_stats(confusion_matrix),
            "keys": self._available_keys,
            "predictions": self._available_predictions,
        }
        for stat in ["total", "correct", "accuracy"]:
            self.summary[f"classwise_{stat}"] = {
                key: key_stats[stat]
                for key, key_stats in self.summary["classwise_stats"].items()
            }
        if field is not None:
            return self.summary[field]
        else:
            return self.summary

    def _compute_accuracy(self):
        return sum(
            prediction == target
            for prediction, target in zip(self.predictions, self.targets)
        ) / len(self.ids)

    def _build_lookups(self):
        self._available_keys = self._get_keys()
        self._available_predictions = list(
            sorted(set(prediction for prediction in self.predictions))
        )
        self._keys_lookup = self._index_lookup(self._available_keys)
        self._predictions_lookup = self._index_lookup(
            self._available_predictions
        )

    def _compute_confusion_matrix(self):
        confusion_matrix = torch.zeros(
            len(self._available_keys), len(self._available_predictions)
        )
        for key, prediction in self._get_confusion_entries():
            key_idx = self._keys_lookup[key]
            prediction_idx = self._predictions_lookup[prediction]
            confusion_matrix[key_idx, prediction_idx] += 1
        return confusion_matrix

    def _compute_classwise_stats(self, confusion_matrix):
        total = confusion_matrix.sum(dim=-1)

        # This can be used with "classes" that are not
        # statically determined; for example, they could
        # be constructed from seq2seq predictions. As a
        # result, one cannot use the diagonal
        key_targets = (
            self._available_keys
            if not self.categories
            else [target for _, target in self._available_keys]
        )
        correct = torch.tensor(
            [
                (
                    confusion_matrix[idx, self._predictions_lookup[target]]
                    if target in self._predictions_lookup
                    else 0
                )
                for idx, target in enumerate(key_targets)
            ]
        )
        accuracy = correct / total
        return {
            key: {
                "total": item_total.item(),
                "correct": item_correct.item(),
                "accuracy": item_accuracy.item(),
            }
            for key, item_total, item_correct, item_accuracy in zip(
                self._available_keys, total, correct, accuracy
            )
        }

    def _get_keys(self):
        if self.categories:
            keys = zip(self.categories, self.targets)
        else:
            keys = self.targets
        return list(sorted(set(keys)))

    def _get_confusion_entries(self):
        if self.categories:
            result = (
                ((category, target), prediction)
                for category, target, prediction in zip(
                    self.categories, self.targets, self.predictions
                )
            )
        else:
            result = zip(self.targets, self.predictions)
        result = list(result)
        return result

    def _index_lookup(self, items):
        return {item: idx for idx, item in enumerate(items)}

    def clear(self):
        """Clears the collected statistics"""
        self.ids = []
        self.predictions = []
        self.targets = []
        self.categories = []

    def write_stats(self, filestream):
        """Outputs the stats to the specified filestream in a human-readable format
        Arguments
        ---------
        filestream: file
            a file-like object
        """
        if self.summary is None:
            self.summarize()
        print(
            f"Overall Accuracy: {self.summary['accuracy']:.0%}", file=filestream
        )
        print(file=filestream)
        self._write_classwise_stats(filestream)
        print(file=filestream)
        self._write_confusion(filestream)

    def _write_classwise_stats(self, filestream):
        self._write_header("Class-Wise Accuracy", filestream=filestream)
        key_labels = {
            key: self._format_key_label(key) for key in self._available_keys
        }
        longest_key_label = max(len(label) for label in key_labels.values())
        for key in self._available_keys:
            stats = self.summary["classwise_stats"][key]
            padded_label = self._pad_to_length(
                self._format_key_label(key), longest_key_label
            )
            print(
                f"{padded_label}: {int(stats['correct'])} / {int(stats['total'])} ({stats['accuracy']:.2%})",
                file=filestream,
            )

    def _write_confusion(self, filestream):
        self._write_header("Confusion", filestream=filestream)
        longest_prediction = max(
            len(prediction) for prediction in self._available_predictions
        )
        confusion_matrix = self.summary["confusion_matrix"].int()
        totals = confusion_matrix.sum(dim=-1)
        for key, key_predictions, total in zip(
            self._available_keys, confusion_matrix, totals
        ):
            target_label = self._format_key_label(key)
            print(f"Target: {target_label}", file=filestream)
            (indexes,) = torch.where(key_predictions > 0)
            total = total.item()
            for index in indexes:
                count = key_predictions[index].item()
                prediction = self._available_predictions[index]
                padded_label = self._pad_to_length(
                    prediction, longest_prediction
                )
                print(
                    f"  -> {padded_label}: {count} / {total} ({count / total:.2%})",
                    file=filestream,
                )

    def _write_header(self, header, filestream):
        print(header, file=filestream)
        print("-" * len(header), file=filestream)

    def _pad_to_length(self, label, length):
        padding = max(0, length - len(label))
        return label + (" " * padding)

    def _format_key_label(self, key):
        if self.categories:
            category, target = key
            label = f"{category} -> {target}"
        else:
            label = key
        return label