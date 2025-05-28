# prompt templates accorrding to paper: https://arxiv.org/html/2402.14382v2#bib

PICK_N_HIS = """
There is a given text consisting of multiple historical events in the form of “{{id}}:[{{subject}} {{relation}} {{object}}{{time}}];”. And there is a query in the form of: “{{subject}} {{relation}} {{whom}} {{time}}?” If you must infer several {{object}} that you think may be the answer to the given query based on the given historical events, what important historical events do you base your predictions on? Please list the top {top_n} most important histories and output their {{id}}.
Here are the given historical events:
{histories}
Here is the query:
{query}

Please only output the {top_n} MOST RELEVANT IDs of the historical events that your inferred answers are based on, without any additional explanations. The output should be a comma-separated list of IDs, and do not repeat any ID. Please strictly follow the format and instructions provided.
"""

PICK_N_CHAINS = """
There is a given text consisting of multiple history chains in the form of “{{id}}:[{{subject}} {{relation}} {{object}} {{time}}, {{subject}} {{relation}} {{object}} {{time}}, ...];”. And there is a query in the form of: “{{subject}} {{relation}} {{whom}} {{time}}?” If you must infer several {{object}} that you think may be the answer to the given query based on the given historical events, what important history chains do you base your predictions on? Please list the top {top_n} most important history chains and output their {{id}}.
Here are the given history chains:
{chains}
Here is the query:
{query}

Please only output the {top_n} MOST RELEVANT IDs of the historical chains that your inferred answers are based on, without any additional explanations. The output should be a comma-separated list of IDs, and do not repeat any ID. Please strictly follow the format and instructions provided.
"""

HIS_TO_ANSWER = """
You must be able to correctly predict the {{whom}} of the given query from a given text consisting
of multiple historical events in the form of “{{subject}} {{relation}} {{object}} {{time}}” and the query in
the form of “{{subject}} {{relation}} {{whom}} {{time}}?” You must output several {{object}} that you
think may be the answer to the given query based on the given historical events. Please list all
possible {{object}} which may be answers to the query. Please assign each answer a serial number
to represent its probability of being the correct answer. Note that answers with a high probability
of being correct should be listed first.
Here are the given historical events:
{histories}
Here is the query:
{query}
lease list all possible {{object}} which may be answers (one per line, No repetition) without explanations.
Note that answers with high probability should be listed first.
For example:
\"\"\"
Possible answers:
1. XXX
2. XXX
3. XXX
· · · · · ·
\"\"\"
Please strictly follow the above demands for output.
"""



