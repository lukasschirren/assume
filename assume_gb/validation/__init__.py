# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validation of simulated GB day-ahead prices against observed prices: a fixed scorecard of eight
criteria and a standard set of figures. The choices follow the synthesis note "How should I
validate an agent-based electricity price model?" (7 Oct 2026); every public function names the
reference it follows by its citation key in that note.

    criterion                                  module
    1  How large are the errors?               metrics      MAE, RMSE, rMAE
    2  Why are they large?                     metrics      bias, sigma ratio, Pearson r, the
                                                            MSE decomposition, daily and weekly
                                                            profiles
    3  Is the distribution right?              metrics      Wasserstein-1, negative prices, tails
    4  Is the model better than simpler ones?  stats        Diebold-Mariano tests
    5  Are the mechanisms right?               mechanisms   one regression on observed and
                                                            simulated prices, binned wind and
                                                            solar effects
    6  Is market power plausible?              mechanisms   markups by scarcity
    7  Does the error matter for decisions?    value        capture prices, storage arbitrage
    8  Is the seed spread informative?         metrics      CRPS and PIT across seeds

``contract`` holds the data contract (``ValidationData``: windows, information sets, the common
sample) and the calendar every criterion shares; ``reference`` the causal estimates criterion 5 is
compared with, read from the package's own ``data/reference``; ``scorecard`` gathers the criteria
into one table and ``figures`` draws them in the style of ``style``. All of these take pandas
objects and read no other files. ``adapters`` is the one module that knows the folders of
``assume_gb`` and turns its runs into the data contract, so everything else can leave this local
folder unchanged; ``report`` writes scorecards, tables and figures into one folder. README.md
beside this file explains the reasoning and how to validate in this repository.

    python -m assume_gb validate --year 2023 --arm learning_novdec_own_e100     the whole validation
    python -m pytest assume_gb/validation          the tests, on synthetic data
"""
