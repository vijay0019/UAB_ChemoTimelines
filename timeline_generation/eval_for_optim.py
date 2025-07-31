from collections import defaultdict
from datetime import datetime
import dateutil.parser


class Chemo:
	def __init__(self, text, first_start=None, last_end=None, cui=None):
		self.text = text
		self.first_start = first_start
		self.last_end = last_end
		self.cui = cui

	def __str__(self):
		return "\t".join(
			[self.text if self.text else "Null", self.cui if self.cui else "Null"]
		)


def relaxed_rel_eval(incorrect, missing, preds, golds):
	not_truly_incorrect = []
	not_truly_missing = []
	for ptup in incorrect:
		is_not_truly_incorrect = False
		chemo, rel, timex = ptup
		# Basically we think contains-1 can be replaced by begins-on/ends-on,
		# and begins-on/ends-on can be replaced by contains-1.
		if rel in ["begins-on", "ends-on"]:
			if [chemo, "contains-1", timex] in golds:
				is_not_truly_incorrect = True
		elif rel == "contains-1":
			if [chemo, "begins-on", timex] in golds or [
				chemo,
				"ends-on",
				timex,
			] in golds:
				is_not_truly_incorrect = True
		if is_not_truly_incorrect:
			not_truly_incorrect.append(ptup)

	for gtup in missing:
		is_not_truly_missing = False
		chemo, rel, timex = gtup

		if rel in ["begins-on", "ends-on"]:
			if [chemo, "contains-1", timex] in preds:
				is_not_truly_missing = True
		elif rel == "contains-1":
			if [chemo, "begins-on", timex] in preds or [
				chemo,
				"ends-on",
				timex,
			] in preds:
				is_not_truly_missing = True
		if is_not_truly_missing:
			not_truly_missing.append(gtup)
	return not_truly_incorrect, not_truly_missing


def relaxed_within_range_eval(incorrect, missing, gold_chemos, pred_chemos):
	"""
		incorrect: false positive,
		missing: false negative,
		gold_chemos and pred_chemos, basically use Chemo object to get the start and end dates for each chemo
	"""
	not_truly_incorrect = []
	not_truly_missing = []
	for ptup in incorrect:
		is_not_truly_incorrect = False
		source, rel, target = ptup

		target = target.replace("w", "W")
		if "W" in target:
			target = datetime.strptime(target + "-1", "%Y-W%W-%w")
		else:
			target = dateutil.parser.parse(target)

		if source in gold_chemos:
			gold_start, gold_end = (
				gold_chemos[source].first_start,
				gold_chemos[source].last_end,
			)
			if not gold_start or not gold_end:
				continue
			if rel in ["ENDS-ON", "ends-on"]:
				# The end date predicted by the system (target), is before the gold start date, then it's wrong.
				if target <= gold_start:
					continue
			if rel in ["BEGINS-ON", "begins-on"]:
				# The start date predicted by the system (target) is after the gold end date, then it's wrong.
				if target >= gold_end:
					continue
			# If the predicted date is in between the gold start and end date, i.e. in the correct range,
			# we consider this is correct, not truly false positive.
			if gold_start <= target <= gold_end:
				is_not_truly_incorrect = True
		if is_not_truly_incorrect:
			not_truly_incorrect.append(ptup)

	for gtup in missing:
		is_not_truly_missing = False
		source, rel, target = gtup

		target = target.replace("w", "W")
		if "W" in target:
			target = datetime.strptime(target + "-1", "%Y-W%W-%w")
		else:
			target = dateutil.parser.parse(target)

		if source in pred_chemos:
			pred_start, pred_end = (
				pred_chemos[source].first_start,
				pred_chemos[source].last_end,
			)
			if not pred_start or not pred_end:
				continue
			if rel in ["ENDS-ON", "ends-on"]:
				if target <= pred_start:
					continue
			if rel in ["BEGINS-ON", "begins-on"]:
				if target >= pred_end:
					continue
			# This is saying, for example, <taxol, contains-1, 2011-03-01> is missing in predictions, i.e. is false negative,
			# however, we can find <taxol, begins-on, 2011-01-01> and <taxol, ends-on, 2011-05-31> in predictions,
			# we consider <taxol, contains-1, 2011-03-01> is not false negative, because the system predicted the
			# correct range that covers the gold timeline.
			if pred_start <= target <= pred_end:
				is_not_truly_missing = True
		if is_not_truly_missing:
			not_truly_missing.append(gtup)
	return not_truly_incorrect, not_truly_missing


def group_chemo_dates(golds):
	gold_group_by_start_end = defaultdict(lambda: defaultdict(list))
	for tup in golds:
		source, label, target = tup
		if label.upper() not in ["BEGINS-ON", "ENDS-ON"]:
			continue
		target = target.replace("w", "W")
		if "-W" in target:
			target = datetime.strptime(target + "-1", "%Y-W%W-%w")
		else:
			target = dateutil.parser.parse(target)

		gold_group_by_start_end[source][label].append(target)
	all_gold_chemos = {}
	# all_gold_chemos: maps from the text of this chemo to the chemo event obj
	for chemo, labels in gold_group_by_start_end.items():
		first_date = (
			min(labels["BEGINS-ON".lower()]) if "BEGINS-ON".lower() in labels else None
		)
		last_date = (
			max(labels["ENDS-ON".lower()]) if "ENDS-ON".lower() in labels else None
		)
		# For each chemo, find the earliest start date, and the latest end date,
		# use them to get the span of this chemo, the span will be used when doing relaxed evaluation.
		chemo_event = Chemo(text=chemo, first_start=first_date, last_end=last_date)
		all_gold_chemos[chemo] = chemo_event
	return all_gold_chemos


def normalize_to_month_and_year(golds):
	month_only_pairs = []
	year_only_pairs = []
	for tup in golds:
		source, label, target = tup
		target = target.replace("w", "W")
		if "W" in target:
			target = datetime.strptime(target + "-1", "%Y-W%W-%w")
		else:
			target = dateutil.parser.parse(target)
		year = target.year
		month = target.month
		if month < 10:
			normalized_month = str(year) + "-0" + str(month)
		else:
			normalized_month = str(year) + "-" + str(month)
		normalized_year = str(year)

		month_pair = [source, label, normalized_month]
		year_pair = [source, label, normalized_year]

		if month_pair not in month_only_pairs:
			month_only_pairs.append(month_pair)
		if year_pair not in year_only_pairs:
			year_only_pairs.append(year_pair)

	has_more_specific_chemos_month = summarize_timeline(month_only_pairs)
	has_more_specific_chemos_year = summarize_timeline(year_only_pairs)
	month_only_pairs = [
		tup for tup in month_only_pairs if tup not in has_more_specific_chemos_month
	]
	year_only_pairs = [
		tup for tup in year_only_pairs if tup not in has_more_specific_chemos_year
	]
	return month_only_pairs, year_only_pairs


def summarize_timeline(timelines):
	"""
		This is to postprocess timelines one more time after we normalized original timeline
		to year only or month only timelines. What it does is: if we have a generic chemo mention,
		e.g. <chemotherapy, contains-1, 2011-01>, or <chemoradiation, contains-1, 2011-01>, we want to see
		if we can have more specific chemo mention happened on the same date with the same label,
		e.g. <Taxol, contains-1, 2011-01>. If we find a more specific chemo mention,
		we would ignore the generic chemo mention, only add <Taxol, contains-1, 2011-01> to the timeline.
	"""
	date_rel_to_chemo = defaultdict(lambda: defaultdict(list))
	for tup in timelines:
		chemo, rel, date = tup
		date_rel_to_chemo[date][rel].append(chemo)

	has_more_specific_chemos = []
	for date, rel_chemos in date_rel_to_chemo.items():
		for rel, chemos in rel_chemos.items():
			for chemo in chemos:
				# chemo.startswith("chemo") is how we check if this is a generic chemo mention
				if chemo.startswith("chemo"):
					if len(date_rel_to_chemo[date][rel]) > 1:
						has_more_specific_chemos.append([chemo, rel, date])
	return has_more_specific_chemos


def strict_eval(gold, pred):
	true_pos = [prediction for prediction in pred if prediction in gold]
	false_pos = [prediction for prediction in pred if prediction not in gold]
	false_neg = [correct for correct in gold if correct not in pred]
	not_truly_fp = fp_fn_single_count(false_pos, false_neg)
	false_pos = [pred for pred in false_pos if pred not in not_truly_fp]
	return true_pos, false_pos, false_neg


def fp_fn_single_count(false_pos, false_neg):
	"""
		What it does here is: let's say in pred we have <Taxol, BEGINS-ON, 2011-01-01>,
		in gold we have <Taxol, CONTAINS-1, 2011-01-01>, then <Taxol, BEGINS-ON, 2011-01-01> would be false positive,
		<Taxol, CONTAINS-1, 2011-01-01> would be false negative, that means, the same mistake is counted twice,
		once in fp, once in fn. So, here, we want to make sure, we count <Taxol, CONTAINS-1, 2011-01-01> as false negative,
		and don't count <Taxol, BEGINS-ON, 2011-01-01> as false positive.
	"""
	not_truly_fp = []
	# false_neg_tracker: (chemo, timex) to label
	false_neg_tracker = {(item[0], item[-1]): item[1] for item in false_neg}
	for ptup in false_pos:
		# E.g. we check in <Taxol, 2011-01-01> is already in false negative.
		if (ptup[0], ptup[-1]) in false_neg_tracker:
			not_truly_fp.append(ptup)
	return not_truly_fp


def relaxed_eval(gold, gold_chemo, pred, pred_chemo):
	true_pos = [prediction for prediction in pred if prediction in gold]
	false_pos = [prediction for prediction in pred if prediction not in gold]
	false_neg = [correct for correct in gold if correct not in pred]
	not_truly_fp_with_range, not_truly_fn_with_range = relaxed_within_range_eval(
		incorrect=false_pos,
		missing=false_neg,
		gold_chemos=gold_chemo,
		pred_chemos=pred_chemo,
	)
	not_truly_fp_with_label, not_truly_fn_with_label = relaxed_rel_eval(
		incorrect=false_pos, missing=false_neg, preds=pred, golds=gold
	)

	not_truly_fp_as_label_single_count = fp_fn_single_count(false_pos, false_neg)

	truly_tp = true_pos
	truly_fp, truly_fn = [], []
	for tup in false_pos:
		if tup in not_truly_fp_with_range or tup in not_truly_fp_with_label:
			# Add this one to true positive if it's not considered as true fp
			truly_tp.append(tup)
		elif tup in not_truly_fp_as_label_single_count:
			continue
		else:
			truly_fp.append(tup)
	for tup in false_neg:
		if tup in not_truly_fn_with_range or tup in not_truly_fn_with_label:
			continue
		truly_fn.append(tup)
	return (
		truly_tp,
		truly_fp,
		truly_fn,
		not_truly_fp_with_range,
		not_truly_fn_with_range,
		not_truly_fp_with_label,
		not_truly_fn_with_label,
	)


def evaluation_f1(gold, pred, strict=True, relaxed_to="day"):
	"""
	Comprehensive evaluation function adapted from the eval script.
	Returns F1 score using the same logic as the official evaluation.
	"""
	# Convert timeline tuples to lists for compatibility
	gold = [list(item) for item in gold]
	pred = [list(item) for item in pred]

	# Get the earliest start and latest end dates for each chemo
	all_gold_chemos = group_chemo_dates(gold)
	all_pred_chemos = group_chemo_dates(pred)

	gold_month_timeline, gold_year_timeline = normalize_to_month_and_year(gold)
	pred_month_timeline, pred_year_timeline = normalize_to_month_and_year(pred)

	if strict:
		true_pos, false_pos, false_neg = strict_eval(gold, pred)
	else:
		if relaxed_to == "day":
			(
				true_pos,
				false_pos,
				false_neg,
				rmv_from_fp_range,
				rmv_from_fn_range,
				rmv_from_fp_label,
				rmv_from_fn_label,
			) = relaxed_eval(gold, all_gold_chemos, pred, all_pred_chemos)
		elif relaxed_to == "month":
			(
				true_pos,
				false_pos,
				false_neg,
				rmv_from_fp_range,
				rmv_from_fn_range,
				rmv_from_fp_label,
				rmv_from_fn_label,
			) = relaxed_eval(
				gold_month_timeline,
				all_gold_chemos,
				pred_month_timeline,
				all_pred_chemos,
			)
		elif relaxed_to == "year":
			(
				true_pos,
				false_pos,
				false_neg,
				rmv_from_fp_range,
				rmv_from_fn_range,
				rmv_from_fp_label,
				rmv_from_fn_label,
			) = relaxed_eval(
				gold_year_timeline, all_gold_chemos, pred_year_timeline, all_pred_chemos
			)
		else:
			raise ValueError("--relaxed_to must be one of 'day', 'month', or 'year'")

	if len(true_pos) + len(false_neg) == 0:
		precision, recall, f1 = 0, 0, 0
	elif len(true_pos) + len(false_pos) == 0:
		precision, recall, f1 = 0, 0, 0
	else:
		precision = len(true_pos) / (len(true_pos) + len(false_pos))
		recall = len(true_pos) / (len(true_pos) + len(false_neg))
		if precision + recall:
			f1 = 2 * (precision * recall) / (precision + recall)
		else:
			f1 = 0

	return f1


