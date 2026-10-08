# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""LOCAL folder, not part of the repository: the GB scenario data and the tool that builds it.
It is listed in ``.git/info/exclude`` and must not be pushed.

The tool reads the offer stack that the merit-order model of the GB data repository exports and
writes ordinary ASSUME scenario folders to ``assume_gb/inputs/``. A scenario folder is
self-contained: the framework runs it with ``-i assume_gb/inputs`` and needs nothing from the other
repository. No model logic lives in this folder; how a supported plant bids is the framework's
``powerplant_energy_naive_support`` strategy.

    python -m assume_gb build --year 2023      offer stack -> assume_gb/inputs/gb_2023
    python -m assume_gb check --year 2023      the scenario's offers against the model's own
    python -m assume_gb run   --year 2023      simulate the scenario and compare its prices
    python -m pytest assume_gb                 the tool's tests and the built scenarios' first weeks
    assume -s gb_2023 -c day_ahead -i assume_gb/inputs -csv <out>     the framework's own CLI

Why the data is here and not in ``examples/inputs``: that folder holds the example scenarios the
framework publishes (tracked, a licence file beside every data file, copied into the Docker
image). Data that stays local goes in a folder of its own, which the loader takes as
``inputs_path``. To publish a scenario later, move its folder to ``examples/inputs`` and add the
licence files; ``observed_prices.csv`` is licensed exchange data and stays out either way.

Modules: ``paths`` (where things are), ``stack`` (reader for the stack export), ``convert``
(stack -> scenario folder), ``compare`` (run a scenario, compare its prices with its references),
``results`` (a run's CSV output read back: positions, generation by group, price setters),
``figures`` and ``style`` (the standard figures of a run), ``viewer`` (the interactive page of a
run), ``learning_figures`` (the figures of a learning run, from its database).
``validation`` is the validation against observed prices, a package of its own (its README).

    python -m assume_gb figures <run folder>   the figures and viewer of a run (see the README)
"""
