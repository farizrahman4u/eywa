from eywa.nlu import Pattern

p = Pattern('[fruit: apple, banana] is my favourite fruit')  # create variable [fruit] with sample values {apple, babana}

p('i like grapes')  # >> {'fruit' : 'grapes'}