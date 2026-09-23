"""One explicit writer contract: the user's request and supplied material only."""

PROTOCOL = "user-only-v1"


def messages(request):
    if not isinstance(request, str) or not request.strip():
        raise ValueError("Writing request must be nonempty text")
    return [{"role": "user", "content": request}]


def judge_state(request, draft):
    # Exactly the user content encoded for the writer; no hidden system rules,
    # report-only checklists, sample metadata, author labels or expected scores.
    return {"REQUEST": messages(request)[0]["content"], "DRAFT": draft}
