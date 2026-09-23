"""New synthetic TASK INPUTS only: twelve training cases and 24 held-out cases.

No desired completions, preferences, historical judge scores or reasoning traces.
These scenarios are fictional; they are not represented as reporting real news.
"""


def case(split, slug, family, request, context, checks, *, grounded=True, shape="source_notes"):
    return {
        "id": f"scale1-{split}-{slug}",
        "scenario_id": f"scale1-{split}-{slug}",
        "source_id": f"scale1-{split}-{slug}",
        "family": family,
        "request": request,
        "context": context,
        "content_checks": checks,
        "grounding_applicable": grounded,
        "input_shape": shape,
    }


TRAIN = [
    case(
        "train",
        "router-trial",
        "social",
        "Make a developer-facing LinkedIn post from these experiment notes.",
        "We tried routing support questions between a small language model and a larger one at the fictional Northline Tools. Dataset: 600 previously resolved questions, with customer details removed. Rules and thresholds were frozen before the replay. Relative to sending every question to the larger model, the routed setup used 34% fewer billed output tokens. Reviewer acceptance was 78% for routing and 81% for the larger-model-only setup. Those are point estimates from one replay, not proof of equivalence. The team has not calculated latency or production savings. Unclear questions and questions mentioning account permissions went to the larger model. We have not deployed the router. Next: a separate fresh set, including questions that mix two issues. We want to explain the trade-off, not announce a product release.",
        [
            "Keep the 34% token reduction separate from unmeasured money/latency savings.",
            "Report the 78% vs 81% acceptance trade-off and no deployment.",
        ],
    ),
    case(
        "train",
        "rlvr-notebook",
        "social",
        "Turn this notebook entry into a tweet thread for ML engineers.",
        "Fictional lab notebook: We trained a code assistant using unit-test outcomes as rewards. The run used 180 short programming exercises. The reward checks whether the submitted function passes the provided tests; it does not inspect elegance, documentation or security. Training-set pass rate increased. We are withholding the exact percentages until rerunning evaluation with a clean test harness. A debugging run found one exercise whose expected answer was wrong. Another allowed a hard-coded answer because its tests never varied the input. Both are removed from the next training set, not retroactively removed from old results. Our takeaway is that a verifier can be automated and still be incomplete. No comparison to human-feedback RL was run.",
        [
            "Explain what test rewards do and do not verify.",
            "Do not invent performance percentages or claim superiority to RLHF.",
        ],
    ),
    case(
        "train",
        "expert-capacity",
        "summary",
        "Explain the finding for a teammate who knows dense models but is new to MoE.",
        "Internal note for an invented architecture experiment: each token selects two of sixteen feed-forward experts. Attention layers are shared. The routing weights are learned, and an auxiliary balancing term discourages concentrating all traffic on a few experts. A routing choice activates only part of the feed-forward parameters, but all expert weights still need storage somewhere in the system. In this experiment, training became less stable when the balancing term was removed. We did not measure serving cost. The note does not claim that experts map cleanly to human concepts or that two active experts mean the whole model costs one eighth as much to run. Communication and attention remain relevant.",
        [
            "Distinguish active feed-forward computation from total weight storage.",
            "Avoid literal specialization or unmeasured cost claims.",
        ],
    ),
    case(
        "train",
        "audio-release",
        "social",
        "Draft a launch post that makes the useful part obvious.",
        "Product: QuietDraft, a fictional desktop recorder for rehearsal notes. Today's 0.4 release lets users mark a moment with the space bar while recording, then jump between marks during playback. Marks have optional text labels. Files and marks stay in a local project folder; there is no cloud sync. The app records audio, not video. Existing recordings can be imported but marks must be added manually. This is a free preview for macOS. Windows is being investigated, with no announced date. Download is via the project website, but no URL has been supplied for this exercise. Do not call it an AI transcription tool: transcription is not a feature.",
        [
            "Name QuietDraft and explain manual audio markers.",
            "Keep local-only, macOS preview and no transcription accurate.",
        ],
    ),
    case(
        "train",
        "inference-batch",
        "rewrite",
        "This sounds like a press release. Rewrite it for our engineering blog.",
        "Draft: We are thrilled to unveil a revolutionary throughput milestone that changes inference forever. By harnessing batching, our next-generation gateway unlocks seamless efficiency for every customer.\nActual experiment: replayed 12,000 short requests against an internal toy endpoint. With a 15 ms queue window, completed requests per second increased from 44 to 59. Median end-to-end latency rose from 210 ms to 224 ms; p95 rose from 480 to 570 ms. All requests used the same model and hardware. The queue window is disabled by default. This replay did not include long-document requests, cancellations or failures. We have not shipped a public feature. The point is to describe a measured throughput/latency trade-off, with the limitations visible.",
        [
            "Keep both throughput improvement and increased median/p95 latency.",
            "Do not claim production launch or all-workload gains.",
        ],
    ),
    case(
        "train",
        "library-seeds",
        "social",
        "Can you make this into a neighborhood Facebook post?",
        "Willow Shelf seed swap notes: Saturday 10–1 in the library courtyard. Free. Bring dry seeds in labeled paper envelopes if you have them; bringing seeds is not required to take part. Label the plant name and year collected, and note if you are unsure of the variety. Volunteers cannot certify germination rates. Please do not bring live plants, soil or commercial seed packets with restrictions on redistribution. Rain location is the covered passage beside the library, not the reading room. Children are welcome with an adult. There is no booking list. We have a few spare envelopes, but ask people to bring their own if possible.",
        [
            "Include time, rain location and participation without a contribution.",
            "Do not guarantee germination or unlimited supplies.",
        ],
    ),
    case(
        "train",
        "cache-ttl",
        "reply",
        "Write my reply to the last message.",
        "Mira: Our dashboard still shows yesterday's total. I cleared the browser cache.\nMe: This page reads from a server-side report cache with a 30-minute TTL. Clearing browser data won't clear that cache.\nMira: It's been more than 30 minutes. Do we just reduce the TTL?\nWhat I checked: the refresh worker has been paused since last night. Its restart is assigned to Lee, who is investigating now. I can run a manual refresh after Lee confirms the worker is safe to resume, but cannot promise a time. The TTL determines when a cached result is stale; it does not itself run the refresh worker. No evidence of missing source records. We should not claim the incident is fixed or tell Mira to keep clearing browser data.",
        [
            "Distinguish TTL expiration from execution of the refresh worker.",
            "Do not promise a resolution time or claim missing data.",
        ],
    ),
    case(
        "train",
        "bike-count",
        "notes",
        "Write the volunteers' update from my scrappy notes.",
        "Saturday repair table: 19 bikes checked, not 19 repaired. 11 small fixes completed. 5 need parts, 3 need a shop with specialist tools. Donations: 74 tokens. New patch kit cost 22; rest stays in the group's tin. Sasha lent the stand and wants it back Tuesday. Two volunteers stayed an extra hour packing up; ask if anyone can help with cleanup next time. No date yet, depends on courtyard permission. One visitor offered an old tool chest; haven't collected it or checked its condition. Don't call it ours yet. Lovely moment: a kid rang the bell every time a wheel spun freely.",
        [
            "Keep checked/repaired counts distinct and next date unconfirmed.",
            "Keep the bell detail; do not invent the child's feelings or a donated chest arrival.",
        ],
    ),
    case(
        "train",
        "support-hours",
        "cut",
        "Shorten this notice, but keep the detail people could get caught by.",
        "From next Monday, our fictional Framefolk workshop's desk will answer phone calls between ten and two, Monday to Thursday. These are phone hours, not workshop opening hours. The workshop stays open for existing booked sessions as before. Outside the phone hours you can leave a voicemail, but we will not check messages during evening sessions. Please include your name and session date. If you need to cancel, cancellation is counted from the time your message is received, not from when we return the call. The existing 24-hour cancellation policy has not changed.",
        [
            "Phone hours are not opening hours.",
            "Cancellation timestamp and unchanged 24-hour policy survive the cut.",
        ],
    ),
    case(
        "train",
        "eval-memo",
        "social",
        "Help me write a candid short post about this lesson from our evaluation.",
        "We compared two versions of our fictional catalog assistant on 80 saved customer questions. The first dashboard reported a six-point increase in answer acceptance. Then we noticed that the new version returned shorter answers and left out delivery exceptions. We added exception coverage as a separate diagnostic, not as a reward, and reviewed all 80 questions again. The acceptance increase did not establish an overall improvement. We have not calculated a final win rate. What we changed operationally: keep all drafts, show lengths alongside scores, and display the original question next to every example. We are sharing a workflow mistake, not releasing a benchmark or accusing another company of cheating.",
        [
            "Do not retain the six-point headline as a verified overall gain.",
            "Explain omissions, length diagnostics and full-draft review concretely.",
        ],
    ),
    case(
        "train",
        "new-job-lab",
        "social",
        "Write my new-job announcement for LinkedIn.",
        "I'm joining the fictional Tern Lab as a developer advocate next month. I'll work on tutorials and example projects for their audio tools. Previously I maintained docs at a small library-software cooperative; the team there helped me learn to explain a bug before offering a workaround. I want to thank that team, not single out invented mentors. Tern Lab is not launching a new product with my arrival. My exact start date and office location aren't public. Warm but not a life-changing-journey speech.",
        [
            "Name the actual role and tutorial/audio focus.",
            "Do not invent employers, mentors, exact start dates or product announcements.",
        ],
    ),
    case(
        "train",
        "label-printer",
        "direct",
        "Write a dryly funny note for the office label printer: it can print labels, but it cannot decide whose lunch this is.",
        "",
        ["Deliver the note itself with the lunch/printer joke, not advice about writing one."],
        grounded=False,
        shape="request_only",
    ),
]


VALIDATION = [
    case(
        "validation",
        "retrieval-release",
        "social",
        "Write a tweet thread announcing this release.",
        "Fictional project Dockleaf 0.7: search results can now show the paragraph that matched, with its file name. This is retrieval, not an answer-writing feature. Snippets come from the imported text; the app does not verify that a source is correct. Index updates are manual: choose Rebuild after editing documents. Plain-text and Markdown files are supported; PDF import is still experimental and tables can be lost. Existing local indexes need rebuilding once after the upgrade. The release is available for Linux and macOS. We have not measured a speed improvement. Downloads are on the project site; no specific link is supplied here.",
        [
            "Name Dockleaf and matched-paragraph snippets, not generated answers.",
            "Include one-time rebuild and manual updates without speed or correctness promises.",
        ],
    ),
    case(
        "validation",
        "model-abstention",
        "social",
        "Turn this evaluation note into a LinkedIn post for AI builders.",
        "At fictional Pinwheel Research, we tested adding an abstain option to a document assistant. Set: 120 questions about a synthetic product manual. Forty questions could not be answered from the manual. With the new prompt, unsupported assertions fell from 18 to 9, but unnecessary refusals on answerable questions increased from 6 to 14. Same model; no training or new retrieval system. Two reviewers checked whether each answer followed the manual; disagreements were resolved together. No latency or cost comparison. This is a small diagnostic, not proof that the prompt generalizes. We are deciding whether to change the instruction, not announcing a solved hallucination problem.",
        [
            "Keep fewer unsupported assertions and more unnecessary refusals together.",
            "Same model/prompt-only experiment; no invented generalization or training result.",
        ],
    ),
    case(
        "validation",
        "orchard-walk",
        "social",
        "Make an inviting community post from these notes.",
        "Mossbridge orchard walk: Sunday 3–4:30, meet at the north gate. Free, with a maximum group of 18; reserve by phoning the volunteer desk. The walk is about pruning decisions, not a hands-on pruning class. The route crosses wet grass and one narrow stile. A shorter step-free route to the first two trees is available if requested when booking, but it does not cover the full tour. No fruit picking during the walk. If high winds are forecast, organizers will call booked visitors on Sunday morning. No alternative date is arranged yet. Dogs cannot enter the orchard, including on leads.",
        [
            "Reserve by phone, max 18; describe access limitation honestly.",
            "No picking, no hands-on lesson, and weather cancellation not rescheduling promise.",
        ],
    ),
    case(
        "validation",
        "funding-pilot",
        "social",
        "Draft our funding announcement. Keep it specific.",
        "Fictional nonprofit Switchyard Stories received a 12,000-credit local arts grant for a six-month oral-history pilot. The funds cover recorder rental, transcription and travel reimbursement for volunteer interviewers. It is not investment and no equity changes hands. Interviewees can choose whether their recordings are kept private or included in a future listening event. There is no event date yet and no public archive has launched. Applications for volunteer interviewers open next Wednesday at the community desk. Prior experience is not required; training is included. The grant doesn't pay a salary to every volunteer. We want to say what the money enables without implying the recordings already exist.",
        [
            "Distinguish a nonprofit grant from an investment round.",
            "No invented launch/date or guaranteed volunteer salaries; consent choice matters.",
        ],
    ),
    case(
        "validation",
        "queue-benchmark",
        "social",
        "Make a developer tweet from this benchmark result.",
        "Fictional Streamlet job queue experiment. 2,400 identical small jobs, one worker process, same laptop. Batched acknowledgments completed the replay in 82 seconds; individual acknowledgments took 105 seconds. Batch size was 16. The run excludes job failures and worker crashes. In the batch variant, up to 15 completed jobs might be repeated after a crash because their acknowledgment wasn't yet written. Job handlers therefore still need to tolerate duplicate execution. We ran this replay three times; times above are medians. We did not test multiple workers or cloud machines. No claim about exactly-once processing.",
        [
            "Mention the crash/duplicate-execution trade-off alongside measured speed.",
            "Do not turn a replay into universal scaling or exactly-once claims.",
        ],
    ),
    case(
        "validation",
        "ceramic-exhibit",
        "social",
        "Use this interview to introduce the exhibition on Instagram.",
        "Fictional interview with potter Ayo Lune. Exhibition title: Cups That Wait. Ayo keeps cups that lean, wobble or have thick rims and arranges them on narrow shelves. They are new pieces made deliberately this way, not rescued factory rejects. Ayo says, 'I wanted the shelf to look as if it was still making up its mind.' Visitors can look but cannot handle these pieces. Show at the Drift Room, Friday through Sunday for three weekends, noon–5. Free entry; no sales at the show. Ayo also makes ordinary usable tableware in a separate practice. This exhibition is not a claim that these uneven pieces are suitable for drinking.",
        [
            "Accurately explain deliberate new work and preserve or paraphrase the real supplied quote.",
            "No handling, sales, recycled-origin or drinkware-safety claim.",
        ],
    ),
    case(
        "validation",
        "garden-night",
        "social",
        "Help me write a cheerful post about how last night went.",
        "We held our first lantern evening in the allotment shed. 27 neighbors came. Everyone brought a mug; we supplied hot apple juice from a borrowed urn. It rained halfway through and we moved the lanterns inside. Three paper lanterns collapsed; Bea turned the frames into tiny coat hooks while we waited. Donations covered apples and extension-cable rental, not the borrowed urn. I promised to return the urn Monday. We'd like to do another evening, but the committee hasn't approved the shed again. No photos of children will be posted. These are personal notes, not a fundraiser.",
        [
            "Keep urn borrowed, donations' actual use and next event undecided.",
            "Retain at least one concrete mishap/detail without fabricated experiences.",
        ],
    ),
    case(
        "validation",
        "release-rollback",
        "social",
        "Write a candid company update for users, using this incident note.",
        "Fictional Bluecap editor: version 2.6 introduced a search bug for files with accented names. We rolled back to 2.5 today. Existing documents were not changed by the bug. Search could omit matching files; it did not delete them. A fix is in review, but no release time is promised. People who installed 2.6 should restart the app to receive the rollback; users on managed devices may need their administrator. The rollback removes the new split-preview feature temporarily. Support wants examples of searches that still miss files after rollback, including app version, but not private document contents.",
        [
            "Explain omitted search results rather than deletion.",
            "Mention restart/admin caveat and temporary feature loss; don't promise a fix date.",
        ],
    ),
    case(
        "validation",
        "mentor-post",
        "social",
        "Write a short thank-you post for my mentor.",
        "Mentor: Len. We worked together on the fictional Grainbox open-source importer for four months. Len taught me to include the smallest failing input in a bug report and to say when I didn't know why a fix worked. My first accepted patch repaired one error message. I still struggle with asynchronous code. This is not a promotion announcement. I'd like the post to feel like something I could actually say, with the error-message patch as its specific memory. No company names, invented late-night calls or claims that Len changed every part of my life.",
        [
            "Deliver a finished thank-you using the small patch and concrete lesson.",
            "No invented promotion, calls or dramatic personal history.",
        ],
    ),
    case(
        "validation",
        "rainwater-trial",
        "social",
        "Turn this report into a neighborhood newsletter post.",
        "Fictional Calmer Court installed two rain barrels at its shared greenhouse in April. During six recorded watering sessions in May, volunteers used collected water for four sessions and tap water for two. Nobody measured liters used, and rainfall was unusually high. The greenhouse roof and gutter were cleaned before installation. The water is for plants only, not drinking or food washing. The trial covers the greenhouse, not all allotment plots. Volunteers must close the lids after use. The committee will inspect the barrels monthly and decide in autumn whether to keep them. No saving on bills has been calculated.",
        [
            "Six sessions/four collected-water sessions, not a quantified percentage of water saved.",
            "Plant-only use and limited wet-month evidence must remain clear.",
        ],
    ),
    case(
        "validation",
        "audio-beta",
        "social",
        "Announce our beta in a useful post, not a list of slogans.",
        "Fictional app SpliceNote lets rehearsal groups attach written comments to timestamps in an uploaded audio file. Version 0.2 beta is open to ten groups by invitation. It does not record live sessions, generate transcripts or score musical performance. Groups can export their own comments as a text file; audio export isn't available. Uploaded audio is stored for 30 days unless the group removes it sooner. Only the group owner can invite members. The beta is free during this trial, with later pricing undecided. People should request an invitation by emailing the project team; the address hasn't been supplied for this writing exercise.",
        [
            "Name SpliceNote and comments on uploaded audio, not transcription.",
            "Retain invite cap, 30-day storage and trial-only free pricing without an invented address.",
        ],
    ),
    case(
        "validation",
        "school-map",
        "social",
        "Write a parent-facing post about the new walking map.",
        "Fictional Harbor Steps school volunteers drew a map of routes families already use. It marks three crossings staffed by the council at arrival time, two steep paths and a gate that opens only during school hours. The map is not a safety certification and routes weren't surveyed by a traffic engineer. Afternoon crossing coverage varies, so families need to check current staffing separately. Printed copies are at reception from Thursday. There is no live tracking app. Volunteers welcome corrections about closures, but parents should not post children's names or daily schedules in the public comments.",
        [
            "Map is descriptive, not a certified safe-route or live-tracking service.",
            "Arrival/afternoon distinction and privacy request retained.",
        ],
    ),
    case(
        "validation",
        "adapter-review",
        "summary",
        "Summarize this for an engineer deciding whether to run a pilot.",
        "A fictional team tested rank-4 adapters on a fixed 9B base model for classifying maintenance tickets. Training used 2,000 labeled tickets from two plants. On a separate set of 300 tickets from those same plants, macro-F1 was 0.82 compared with 0.76 for the unchanged base prompt. Tickets from other plants were not tested. Training took 46 minutes on the team's rented setup; deployment latency wasn't measured. The adapter weights occupy 28 MB, but serving still needs the base model. Three rare ticket categories each had fewer than ten evaluation items. The team recommends a small pilot with local categories and a reviewed holdout, not immediate organization-wide replacement.",
        [
            "Preserve in-domain evaluation and sparse-category limits.",
            "28 MB is adapter size, not the whole deployed model.",
        ],
    ),
    case(
        "validation",
        "museum-returns",
        "summary",
        "Give the front-desk volunteers a clear summary of the change.",
        "Fictional Pebble Museum loan-box policy. From June, borrowers may leave returned activity boxes in the staffed vestibule on Saturdays 10–12 without a prearranged appointment. The box must include its inventory card. Staff will check contents later and email if anything is missing. A drop-off receipt confirms receipt of the box, not that all contents are present. Weekday returns still need an appointment. No outdoor drop box is available. The policy applies to activity boxes, not costumes or display cases. Late fees stop on the recorded receipt date, even if the contents check occurs later. A missing-item charge is a separate assessment after staff contact the borrower.",
        [
            "Distinguish receipt, contents approval and late-fee timing.",
            "Correct scope and Saturday hours; no outdoor drop-off option.",
        ],
    ),
    case(
        "validation",
        "repair-pilot",
        "summary",
        "Write a short blog explaining what we learned from the pilot.",
        "Fictional Fern Row repair counter, eight-week trial: 62 items assessed; 35 repaired in house, 14 referred elsewhere, 13 declined because parts or tools weren't available. We did not track whether referred items were eventually repaired. Visitors paid a 3-credit assessment fee; repair materials cost extra with approval. Volunteers reported that photographs sent ahead helped them prepare, but no measured comparison was made. The counter will run for another eight weeks on the same Saturday schedule. It will not accept mains-powered electrical items. A proposal for Wednesday sessions depends on finding two more trained volunteers and is not approved. We want the post to be useful to someone deciding whether to bring an item.",
        [
            "35 completed repairs, unknown referral outcomes; keep exclusions and material costs.",
            "Saturday extension is approved but Wednesday sessions are not.",
        ],
    ),
    case(
        "validation",
        "policy-rewrite",
        "rewrite",
        "Rewrite this so it sounds straightforward and people can act on it.",
        "Draft from fictional Coverfold studio: We are delighted to announce an exciting evolution in our reservation ecosystem. To empower seamless access and unlock fairness for all, users will now experience a streamlined cancellation journey.\nPolicy: from September 1, cancel a desk booking at least 12 hours before its start for a full credit refund. Later cancellations receive no credit refund, unless the studio closes the room. A closure earns a refund regardless of notice. Existing reservations made before September 1 keep the old six-hour rule. Change applies to desks only; meeting rooms keep their published terms. Cancel from the booking confirmation email, not a social media message.",
        [
            "Preserve grandfathered reservations and studio-closure exception.",
            "Desk-only scope and email cancellation channel retained.",
        ],
    ),
    case(
        "validation",
        "speaker-bio",
        "rewrite",
        "Make this bio sound like a person, keeping the useful details.",
        "Draft: A visionary thought leader at the intersection of innovation and impact, Nessa opens doors to limitless possibility through transformative facilitation.\nFacts: Nessa Rao runs a monthly beginner repair circle at the fictional West Shed. Her background is costume making. She started the circle in 2022 after borrowing a neighbor's sewing machine to mend a coat. At this event she'll demonstrate patching denim by hand. She is not an engineer and has no stated awards. The workshop is for adults who can thread a needle; materials are supplied, but people should bring the garment they want to mend. Write a speaker bio, not a fabricated quote from Nessa.",
        [
            "Accurate repair-circle, costume-making and denim focus.",
            "No invented credentials or awards; retain audience/supplies if including workshop details.",
        ],
    ),
    case(
        "validation",
        "changelog-edit",
        "rewrite",
        "Clean up this release note without removing the awkward limitation.",
        "Fictional Draftkite 1.9 note: We fixed everything about footnotes, delivering an unparalleled experience of effortless writing.\nActually fixed: exporting numbered footnotes from Markdown to HTML no longer duplicates the final note. Nested footnotes are still unsupported. Existing exported HTML files are not repaired automatically; users must export again. The issue affected documents ending with a footnote reference, not all documents. A separate PDF-export problem is still under investigation. The app version is 1.9, not a major redesign. We want the note to tell people whether they should re-export.",
        [
            "Specific duplicated-final-note fix and re-export requirement.",
            "Nested footnotes and PDF issue remain unresolved.",
        ],
    ),
    case(
        "validation",
        "membership-reply",
        "reply",
        "Reply to this member as the club secretary.",
        "Member: I renewed yesterday and the booking form still charges me the visitor rate. Can you refund my ticket for tonight and tell me when it'll be fixed?\nSecretary notes: the payment receipt is visible, but membership access updates overnight. I can manually activate access now after matching the receipt to the member's account. The ticket for tonight was booked last week at the visitor rate. Our written policy doesn't apply a later membership renewal retroactively to earlier tickets. I can ask the treasurer to consider an exception, but cannot approve or promise a refund. Don't imply the member failed to pay. No need to request full card details.",
        [
            "Offer supported manual access correction without blaming payment.",
            "No promised retroactive refund or invented resolution time.",
        ],
    ),
    case(
        "validation",
        "parser-reply",
        "reply",
        "Answer the developer's last question, please.",
        "Fictional PebbleParse issue thread. Dev: The CSV importer turns blank cells into zero. Can I switch this off without rewriting the whole import?\nMaintainer investigation: blank numeric cells currently use the default numeric value. Setting missing_numeric to null preserves them as null. It affects numeric columns only; whitespace-only strings in text columns stay strings. The option arrived in 0.12.2, so the user's 0.12.0 needs upgrading. It doesn't repair previously imported records. Dev: So if I change the flag, our existing dashboard history will fix itself?\nA re-import into a separate table can be tested before changing the dashboard's source. There is no approved one-click migration or guarantee that downstream charts accept null.",
        [
            "No automatic repair; correct minimum version and numeric-only behavior.",
            "Safe test re-import without invented migration or chart compatibility promises.",
        ],
    ),
    case(
        "validation",
        "roof-cut",
        "cut",
        "Cut this down but please don't lose the funny part.",
        "I borrowed a ladder so I could remove the tennis ball from our shed roof. This was a simple task, which is why I first spent twenty minutes looking for shoes and another ten finding somebody to hold the ladder. By the time I climbed up, the ball had rolled off by itself. I came down, returned the ladder, and then noticed the shoes were still on the roof. They are not special shoes. I had taken them off because I didn't want to mark the ladder.",
        [
            "Keep ball retrieving itself and shoes-on-roof reversal.",
            "Shorten rather than explaining the joke or inventing an accident.",
        ],
    ),
    case(
        "validation",
        "festival-notes",
        "notes",
        "Make these notes into our festival volunteers' roundup.",
        "Fictional Canal Pages day: 9 stalls, 3 readings, about 140 visits counted at the gate (not unique visitors). Rain moved the second reading indoors. Jo brought spare chair feet to stop the wobble; applause every time someone sat without tipping. Stall fees covered room hire. We still owe the printer 38 credits. Don't say we broke even. The borrowed banner is back with Ana; microphone still with me, returning Thursday. We'd like volunteer feedback by Monday. Next year's date depends on the venue and hasn't been booked.",
        [
            "Visits not unique people, unpaid printer bill, no break-even claim.",
            "Keep next year unbooked and at least one concrete event detail.",
        ],
    ),
    case(
        "validation",
        "onboarding-job",
        "notes",
        "Use these notes for a low-key post about my first week.",
        "Joined fictional Orbit Loom as a technical writer. First week: learned how their support tags work, fixed two outdated screenshots, spent an hour reproducing a bug that turned out to be my VPN. Colleague Dev patiently found the problem with me. Still learning the product. I am not announcing a promotion or a completed documentation overhaul. I want to thank Dev by first name only. No screenshots or internal metrics can be shared. The interesting part is realizing how much a support ticket assumes the reader already knows.",
        [
            "Small first-week work, VPN mistake and support-ticket lesson.",
            "No invented impact metrics, promotion or overhaul.",
        ],
    ),
    case(
        "validation",
        "museum-sign",
        "direct",
        "Write a gently funny sign for a museum's very squeaky door. Ask people to close it slowly; apologizing to the door is optional.",
        "",
        ["Return a sign with the slow-close request and gentle joke, not multiple writing strategies."],
        grounded=False,
        shape="request_only",
    ),
]
