# banking-products-recommender

<h2>🏦 A recommender system for banking transactions built with python</h2>

This project is my attempt to build recommendation system suitable for banking environment.

Limited by security measures and technology barriers, i managed to develop fully automated proof of concept of a recommender engine, that provides unique insights to bankers.

Each client get a product recommendation based on past behaviour of other clients, who did similar transactions.

E.g. if John Doe, the owner of USD account buys frequently English tea and pays in GBP, engine will pick this behaviour up and recommends him current account, which has cheaper FX conversion Fees.

<h3> Technology stack </h3>

PoC was deployed into production with following architecture:

<b>Data</b>
* Are stored in an enterprise grade Oracle data warehouse. ETL and data cleansing is done by stored procedures.

<b>Backend</b>
* Recommendations are trained and predicted by python instance, that calculates product feasibility using Alternating least square method.

<b>Frontend</b>buys
* Results are displayed to bankers on a dashboard built with PHP, Ajax, Js, HTML stack.

<b>Automation</b>
* Daily traning is achieved by setting a process and data workflow in Microsoft SQL Server Integration Services.

<h3>Lessons learned</h3>

* too much sparse matrix produces poor recommendation
* more transaction history, better results

<h3>Possible future improvements</h3>

* add rule based pre-filtering of transaction types
* generalize transaction types with machine learning clustering methods
