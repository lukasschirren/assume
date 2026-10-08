.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: MIT

######################
Support Policies
######################

Support Policies are a very important feature when considering different energy market designs.
A support policy allows to influence the cash flow of unit, making decisions more profitable.

One can differentiate between support policies which influence the available market capacity (product_type=`energy``) and those which do not.

If the product_type is `energy`, the volume used for the contract can not be additionally bid on the EOM.

All the support policies are only available when using a Market with the MarketMechanism :meth:`assume.markets.clearing_algorithms.contracts.PayAsBidContractRole`


Example Policies
=====================================


Feed-In-Tariff - FIT
--------------------

To create a Feed-In-Tariff (Einspeisevergütung) one has a contract which sets a fixed price for all produced energy.
The energy can not be additionally sold somewhere else (product_type=`energy`).

The Tariff is contracted at the beginning of the simulation and is valid for X days (1 year).

The payout is executed on a different repetition schedule (monthly).
For this, the output_agent is asked how much energy an agent produced in the timeframe.

This is essentially the same as a Power Purchase Agreement (PPA), except that the payment of FIT is continuous and not monthly or yearly.


Fixed Market Premium - MPFIX
----------------------------

A market premium is paid on top of the market results, based on the results.
As the volume does not influcence the market bidding, the product_type is `financial_support`
So a Market premium is contracted at the beginning of the simulation and is valid for X days (1 year).

The payout is executed on a different repetition schedule (monthly).
For this, the output_agent is asked how much energy an agent produced in the timeframe and what the clearing price of the market with name "EOM" was.
The differences are then calculated and paid out on a monthly base.

This mechanism is also known as One-Sided market premium

Variable Market Premium - MPVAR
-------------------------------

The Idea of the variable market premium is to be based on some kind of market index (like ID3) received from the output agent.


Capacity Premium - CP
---------------------

A capacity premium is paid on a yearly basis for a technology.
This is done in € per installed MW of capacity.
It allows to influence the financial flow of plants which would not be profitable.

Contract for Differences - CfD
------------------------------

A fixed LCoE (Levelized Cost of Energy) is set as a price, if an Agent accepts the CfD contract,
it has to bid at the hourly EOM - the difference of the market result is paid/received to/from the contractor.


Swing Contract
--------------

Actor
^^^^^


Bidding with a support contract
=====================================

A support contract also changes the price down to which a unit is willing to generate, as the payments per MWh are lost when it does not run.
The bidding strategy ``powerplant_energy_naive_support`` (:py:meth:`assume.strategies.support_strategies.EnergyNaiveSupportStrategy`) covers this effect on the bids without a contract market.
A power plant bids the lowest price at which generating still pays under its contract, kept within the price limits of the market.

The contract is set by columns of ``powerplant_units.csv``:

========================  ==============================================================================================================
Column                    Description
========================  ==============================================================================================================
support_scheme            ``premium`` or ``cfd``. A unit without a contract (empty) bids its marginal cost.
support_value             The premium or the strike price per MWh.
support_neg_price_rule    For a ``cfd``: ``any_hour`` if the contract pays nothing in any period with a negative price.
support_reference         For a ``cfd``: the column of ``fuel_prices_df.csv`` with the reference price the contract is settled against.
                          Empty if it is settled against the price the unit itself earns.
========================  ==============================================================================================================

The bid price follows from the scheme:

- ``premium``: A fixed payment per MWh on top of the market price, for example a fixed market premium, a generation tariff or a certificate per MWh as under the Renewables Obligation in Great Britain. The unit earns the price plus the premium and bids its marginal cost less the premium.
- ``cfd`` settled against the price the unit itself earns: The contract tops the price up to the strike price, and the top-up is at most the strike price. At a negative price the unit therefore earns the strike price plus that price, and it bids its marginal cost less the strike price. Under the ``any_hour`` rule the contract pays nothing in a period with a negative price, while from a price of zero upwards the unit earns the strike price: it bids zero, or its marginal cost if that is lower. Any other rule, such as a suspension only after six negative hours in a row, is treated as paying in every period.
- ``cfd`` settled against a reference price which the output of the unit does not move, such as a seasonal baseload price: The contract adds the strike price less the reference price to the market price, and the unit bids its marginal cost less this difference.

The strategy only sets the bid price. The payments of the contract are booked by the power plant itself, for every strategy: :py:meth:`assume.units.powerplant.PowerPlant.support_payment` returns what the contract pays for a volume sold at a market price (the premium; the strike price less the market price, or nothing in a negative period under ``any_hour``; the strike price less the reference price), and the plant adds it to ``outputs["support_cashflow"]`` when it books its cashflow. The market cashflow (``energy_cashflow``) stays the market revenue alone, so the two can be told apart in the output.

A plant with a contract can also learn its bids: ``powerplant_energy_learning_support`` (:py:meth:`assume.strategies.learning_support_strategies.EnergyLearningSupportStrategy`) is the reinforcement learning strategy ``powerplant_energy_learning`` with the contract's payments in the income the reward is made of, and with the price down to which generating pays, instead of the marginal cost, as the cost the agent observes and explores around at first. ``portfolio_learning_support`` (:py:meth:`assume.strategies.learning_support_strategies.PortfolioLearningSupportStrategy`) does the same for an operator that bids a portfolio of plants: each plant's inflexible capacity is bid at its contract floor, its flexible capacity at that floor plus the mark-up the agent chooses, and the profit and the competitive benchmark of the reward include the contracts' payments.

A levy on generation revenue
----------------------------

A levy that takes a share of the receipts above a benchmark price, such as the Electricity Generator Levy in Great Britain (45 percent of receipts above 75 GBP/MWh from 2023), is set by two further columns of ``powerplant_units.csv``: ``levy_rate`` (the share) and ``levy_benchmark`` (the price per MWh). :py:meth:`assume.units.powerplant.PowerPlant.levy_payment` returns what the levy takes in a time step and the plant books it to ``outputs["levy_cashflow"]`` (negative) beside its market cashflow; the learning strategies with a support contract count it in their income. A levy of this kind is assessed on a company's realised receipts over a year in practice, so the per-step booking is the marginal view of a plant that sells at the market price: it overstates what a company whose receipts stay below the benchmark on average, or which sold its output forward, would pay.
