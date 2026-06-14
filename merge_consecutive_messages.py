from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    merge_message_runs,
)

messages = [
    SystemMessage("you're a good assistant."),
    SystemMessage("you always respond with a joke."),
    HumanMessage([{"type": "text", "text": "i wonder why it's called langchain"}]),
    AIMessage(
        HumanMessage("and who is harrison chasing anyway"),
        """Well, I guess they thought "WordRope" and "SentenceString" just
didn\'t have the same ring to it!""",
    ),
    AIMessage("""Why, he's probably chasing after the last cup of coffee in the
office!"""),
]
merge_message_runs(messages)
