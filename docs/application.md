# Fraud Hunter — ответы для анкеты

| Поле со списком | Что выбрать |
|---|---|
| Project Name | Fraud Hunter |
| Product Status | MVP |
| HQ Jurisdiction | Kazakhstan |
| Operational City | Almaty — поменяй на свой |
| Industry Sector | FinTech, AI, SaaS |
| Market Type | Global |
| Customer Focus | B2B |

---

## Abstract  (285 / 300 символов)

Fraud Hunter is an anti-fraud engine for banks and fintechs. It catches both stolen-card fraud and social engineering, where the victim pays willingly, then prices all four possible actions in tenge and picks the cheapest. Every decision is explained to analyst, operator and customer.

## Project Description  (2861 / 3000 символов)

Fraud Hunter is a decision engine that sits between a payment request and its authorisation and answers one question: what is the cheapest thing this bank can do with this transaction right now.

THE BLIND SPOT. Classical anti-fraud is built around one story - a stranger pays for the customer. Stolen card, hijacked account, new device, foreign city, VPN. Rules catch that. The dominant fraud in Kazakhstan today is the opposite story: the customer is deceived and pays with their own hands. A call from "the bank", "move your money to a safe account", remote-access software on the phone, a transfer dictated over the line. Own device, own city, no VPN, correct password, correct SMS code. Formally there is nothing to flag, and conventional systems let it through.

Fraud Hunter separates the two, because they need different responses. A push confirmation stops a stranger holding a stolen card 85% of the time - he does not have the customer's phone. Against deception it works 10% of the time: the victim confirms it themselves. Only a pause and a live conversation breaks that scheme. On our test period the system sent 170 of 185 social-engineering cases to an operator call and blocked zero, while blocking 125 of 183 stolen-card cases. Nobody wrote that rule; the cost model derived it.

NO THRESHOLD. Every other system has a probability cut-off, and it is always wrong: too soft for a 3,000,000 KZT transfer, too harsh for a 900 KZT coffee. Fraud Hunter computes the expected loss in tenge for four actions - allow, confirm, hold and call, block - and takes the minimum. Loss given fraud, case handling, operator time, customer friction that scales with the amount, lost margin and churn risk proportional to lifetime value all enter the arithmetic. Business parameters live in a UI, not in code: raise the cost of an operator call from 1,500 to 58,000 KZT and the system stops calling, with no rule touched.

EXPLAINABILITY ON THREE LEVELS. SHAP for the data scientist, plain-language reasons for the operator, and a message for the customer in Russian, Kazakh and English. For a victim that message is the product: "operation suspended for 15 minutes; if someone is calling you right now claiming to be the bank, it is a fraudster; hang up and call us back". Each decision also carries a counterfactual - "had the amount stayed under 129,400 KZT, the decision would have been different".

MEASURED, NOT CLAIMED. Walk-forward validation over four expanding windows: PR-AUC 0.900 (0.876-0.917), 886 fraud cases in test windows. 94.3% of fraud acted upon, friction on 3.41% of honest operations. Losses of 7.1M KZT against 12.3M for the best fixed threshold. 11 ms per decision including the explanation. A second unsupervised channel catches 35.1% of a scheme deliberately erased from training. Drift monitoring gates retraining. 85 automated tests.

## Critical Inefficiency  (583 / 600 символов)

Banks defend against the wrong attacker. Rules and classical ML look for a stranger: new device, foreign country, VPN, abnormal burst. The fraud that now dominates in Kazakhstan produces none of those signals - the customer performs the payment themselves under a fraudster's instructions, from their own phone and city, with the correct password and code. It passes as normal and the money is gone.

The second inefficiency is the fixed threshold. One cut-off serves a 900 KZT coffee and a 3,000,000 KZT transfer, so banks pay twice: in fraud losses and in blocked honest customers.

## Strategic Fix  (545 / 600 символов)

Two changes. First, the engine classifies the scheme, not just the risk, because the counter-measure differs: a push confirmation stops a stolen card 85% of the time and deception 10%, while a live operator call stops deception 80%. Second, we removed the threshold. For each transaction the engine computes the expected loss in tenge of allowing, confirming, holding for a call and blocking, then takes the cheapest.

Every decision carries SHAP values, plain-language reasons, a counterfactual and a message to the customer in three languages.

## Unique Value Proposition  (481 / 500 символов)

The only engine we know of that treats the deceived customer as a separate class of fraud and answers it with a different action rather than a harder block - because a block does not save a victim who is still on the phone with the fraudster.

Decisions are made in tenge, not in probability points, so a risk officer can audit and retune the system without a data scientist. And every decision comes with wording the operator can read aloud, which turns detection into prevention.

## Why Now?  (491 / 500 символов)

Social engineering has overtaken card theft as the main channel of financial fraud in Kazakhstan, and regulators are moving from recommendations to obligations on customer protection and explainable automated decisions. Instant interbank transfers have removed the delay that used to give banks time to intervene.

The technology behind a per-transaction cost decision - calibrated probabilities, SHAP at millisecond latency, streaming features - became commodity only in the last few years.

## Existing Alternatives  (1218 символов)

1. Rule engines inside core banking systems. Fast and transparent, but blind to social engineering by construction: they look for anomalies in device, geography and velocity, and a deceived customer produces none. Every new scheme needs a hand-written rule after the losses have occurred.

2. Vendor anti-fraud platforms. Strong on account takeover and card theft, priced for tier-1 banks, and they answer with a probability plus a threshold. The action is left to the bank, so the "confirm versus call versus block" choice - the decision that determines whether a victim is saved - stays unautomated.

3. In-house ML teams. They produce a good score but rarely a decision policy: the score meets a fixed threshold, the economics are never modelled, and explainability is a dashboard for the data scientist rather than a sentence the operator can say to the customer.

4. Customer education campaigns. Necessary and demonstrably insufficient: the victim is being manipulated in real time, and awareness collapses under pressure from a caller who sounds official.

Fraud Hunter sits in that gap: scheme-aware detection, an action chosen by expected loss in tenge, and an explanation aimed at the person being defrauded.
