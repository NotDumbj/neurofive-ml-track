# Case Study: Flight Delay Risk Estimator

**The problem.** Travelers booking a flight see a carrier's historical
on-time percentage at best — a single number that ignores the specific
route, date, season, and time of day they're actually flying. Delays are
expensive: missed connections, wasted vacation time, and for airlines,
downstream schedule disruption. I wanted to know whether route- and
schedule-level historical data alone could produce a more personalized,
actionable risk signal.

**The approach.** Using ~150,000 US domestic flight records, I engineered
features capturing *when* (month, day of week, departure hour, peak
season) and *where* (origin, destination, a smoothed historical route
delay rate, and distance tier) a flight operates, then benchmarked
Logistic Regression, Random Forest, and XGBoost classifiers inside a single
scikit-learn pipeline, with class weighting to handle the ~82/18 imbalance
between on-time and delayed flights (delay defined per the FAA standard:
arrival 15+ minutes late).

**The results.** The best model (selected by ROC-AUC) reached **0.614
ROC-AUC** and **48% recall** at a delayed-flight base rate of 18.4%. That's
a real but modest signal — meaningfully better than guessing, but a long
way from precise, because the dataset has no weather, air traffic control,
or mechanical data, which are the biggest real-world drivers of delays.

**The value.** Rather than oversell accuracy the data can't support, I
packaged this as a *risk-flagging* tool: a Streamlit app returns a delay
probability and a Low/Medium/High risk bucket for any airline, route, date,
and departure hour combination, trained on the top 20 airports and top 10
carriers by volume. Framed honestly, even a modest signal like this has
real value — it could nudge a traveler to book more buffer time on a
high-risk itinerary, or feed as one input into a larger travel-insurance or
airline-scheduling model. The project's biggest lesson was as much about
scoping and honest communication of model limitations as it was about the
modeling itself: knowing what a 0.61 AUC model is and isn't good for is as
important as building it.
