"""Talk demo: do examples carry more conventions than rules? (toy text-to-SQL context)

``scenario``  the data model (conventions, examples, questions, decision points)
``sqlcheck``  SQL extraction + convention checks on the sqlglot AST
``prompt``    the one place that assembles the system and user text (import it, do not copy it)
``runners``   model-agnostic runner interface, the Claude runner, the disk cache
``experiments``  E1 / E2 / E3 cells, scoring, aggregation
"""
