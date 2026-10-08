<!--
SPDX-FileCopyrightText: ASSUME Developers

SPDX-License-Identifier: MIT
-->

# Ownership in the GB scenarios

Who owns each unit, who runs it and who trades it, year by year, for the agents of the GB scenarios. Written on 6
October 2026; it replaces the hand-written tables of `owners.py`. Since the evening of 6 October the registers,
the curated tables (each row with its sources) and the rules live in **gb-power-data**, the public GB data
package (the sibling folder `../gb-power-data` or `GB_POWER_DATA_REPO`; its `docs/ownership.md` describes them
and `docs/ownership_references.md` lists every curated share with its evidence). `ownership.py` here keeps only
what belongs to the model: which station, farm or contract each stack unit is, the split of merged scenario
units, and the battery fleet as one unit.

```bash
# in gb-power-data (its config.local.yaml points its raw folder at the GB repository's data/raw)
python -m gbpowerdata.build.bm_units --years 2018 2026          # BM units by year, activity dated by output
python -m gbpowerdata.build.build_ownership --years 2018 2026   # ownership_{year}.csv, ownership_bmu_{year}.csv
# here
python -m assume_gb ownership --years 2018 2025   # roles by year -> ownership_data/built, <scenario>/unit_owners.csv
python -m assume_gb build --year 2023 ...         # writes the scenario's unit_owners.csv as well
python -m assume_gb build --year 2023 --battery-traders   # + case day_ahead_storage_traders
python -m assume_gb agents --years 2018 2025      # the learning agents of built folders, rewritten (trader level)
python -m assume_gb agents --years 2018 2025 --portfolio-cfd   # + cases learning_cfd*: CfD units in the portfolios
python -m pytest assume_gb/test_ownership.py
```

## 1. Three roles

A unit has three kinds of company behind it, and they differ more often than one would think. All three are
taken at the end of each year (the current year: today).

| Role | What it is | Source (in gb-power-data) | What the scenarios use it for |
| --- | --- | --- | --- |
| owner | holds the equity, with shares where a source gives them | DESNZ's list of power stations (DUKES 5.11, one edition per year), Ofgem's RO register, LCCC's CfD register, NESO's Capacity Market register, the curated joint ventures and sales | whose profit it is: owner-weighted profits, policy incidence, portfolio size |
| operator | runs the unit for its owners | the owner if there is one; a joint venture's operator from `company_owners.csv` | the generator levy by group (`egl`: a joint venture's whole levy goes to its operator; `owners.owners` returns it); the portfolio agents with `--agent-level operator`; the bidder of a unit whose lead party is its own project company |
| trader | registered to trade the unit in the Balancing Mechanism (its BM lead party) in the year: it submits the unit's notifications and trades its output, and under an offtake contract bears its price | Elexon's register of BM units by year (`ownership_bmu_{year}.csv`) | who bids: the `unit_operator` of the portfolio learning cases (`owners.bidders`, the default since 7 Oct 2026) and, for batteries, of case `day_ahead_storage_traders` |

For a gas station the three are one company (Pembroke: RWE, RWE, RWE Generation UK). For a joint venture they
split (Saltend in 2023: owned 50/50 by SSE and Equinor, run and traded by their company Triton Power). For a
battery they split most: in 2023 the batteries active in the BM are traded by 15 companies, led by Tesla (26% of
their capacity) and EDF (22%), while the batteries with a capacity agreement in force in delivery year 2023/24
belong to 47 owners, led by Gresham House (20%), HEIT (8%), Statera (8%), SMS (6%), Zenobe (4%) and SUSI
Partners (4%); HEIT's companies are Foresight's from 17 June 2025, when Foresight took the trust private. The two
lists cover different fleets (BM-active against CM-contracted), so they are compared as shares.

The agents (the author, 7 Oct 2026): the model evaluates market power, which is exercised by whoever bids a
unit, so `unit_operator` follows the trader for plants and wind farms, as it has for batteries since 6 Oct. The
bidding agent of a unit (`ownership.bidders_from_roles`) is its trader; its operator where none of its BM units
was active in the year, where the trader is not resolved, or where the lead party resolves to itself (no rule,
source or Companies House parent places it), which is then taken as the unit's own project company.
`build --agent-level operator` and `agents --agent-level operator` keep the operators. CfD units stay out of the
portfolios at either level in the cases `learning`, `learning_week` and `learning_year`: their revenue does not
move with the price, so they add nothing to a portfolio's reason to raise it, and they bid their contract as in the
day-ahead case. As an option (the author, 7 Oct 2026: a portfolio that holds its CfD farms may trade differently),
`--portfolio-cfd` (on `build` or `agents`) adds the cases `learning_cfd`, `learning_cfd_week` and
`learning_cfd_year`, in which the CfD units belong to their companies' portfolios
(`convert.write_cfd_variant`). The build merges CfD contracts that offer on the same terms, across companies
(Walney Extension and Burbo Bank Extension, Orsted's, with Dudgeon, Equinor's), so these cases read their own
files with the merged CfD units split back into contracts: `powerplant_units_learning_cfd.csv`,
`unit_operators_learning_cfd.csv` and `availability_df_cfd.csv.gz`. The offers are the same, so without learning
both sets of cases clear at the same prices (`test_convert.py`); the year case loads the policies of
`learning_cfd`, and `hpc/make_cases.sh` builds a campaign on them with `SOURCE=learning_cfd`.

## 2. How it is built

**Units.** The scenario's units come from the merit-order model's stack (`comb_stack_agents_{year}_*`), and
`unit_columns.csv` lists the stack units behind each scenario unit. Four kinds are attributed:

- named thermal plants (`Pembroke`, `Keadby`, ...): `curated/plants.csv` names each plant's station in
  gb-power-data's `dukes_bmu_map.csv` (the DUKES site pattern, the technology class and the BM units); the owner
  is the company DESNZ lists for the station (the edition of the following May for the end of the year, the
  edition before for a plant that closed in between, the latest list that has it for a plant DESNZ leaves out
  while its BM units run: Sutton Bridge in 2022-2024; DESNZ names Grain "Grain CHP" in 2018-2020), resolved with
  the station's dated rows first;
- named RO wind farms (`FARM_...`): the farm map's station (the stack's name of the unit, or its id), its BM
  units and the holder of the RO stations its name matches (`build_ownership.farm_ro_links`); where that holder
  has no dated rows but the joint venture that trades the farm's BM units has, gb-power-data's BM-unit table gives
  the owners (Gunfleet Sands);
- CfD units (contract ids such as `INV-HOR-001`): the counterparty in LCCC's register, BM units from LCCC's
  mapping for the year;
- the battery fleet: traders from gb-power-data's BM-unit table (the battery units active in the year, by their
  trader's group), owners from its register table (the battery units of the CM register with an agreement in
  force in the delivery year, by owner group).

Everything else (the model's merged classes of small plants, imports) has no owner of record: `unresolved`.

**Companies to groups** (`gbpowerdata.common.companies.resolve`), first match wins:

1. `company_owners.csv`: dated rows for a station (`station:Saltend`) or a company (`company:Seabank Power
   Limited`, and the register's other spellings of it through `same_as`): owners with shares, the operator,
   `valid_from`/`valid_to`, sources, evidence, `status` (verified, partly verified, to verify, judgement);
2. `company_groups.csv`: a name to its group (DESNZ's labels such as "RWE Npower", project companies such as
   "Keadby Generation Limited");
3. the group's name in the company's name ("RWE Renewables UK Humber Wind"), then in its registered address, then
   the site of its registered address;
4. its majority controller on the date in Companies House's register of persons with significant control, where
   gb-power-data has pulled it and the chain reaches a group known by alias or name (`ch_psc`; dated, so before the
   CM parent). Pulled on 7 Oct 2026 for the CfD project companies and for the 50 BM lead parties that resolved to
   themselves (Kilbraur Wind Energy: Renantis; Aberdeen Offshore Wind Farm: Vattenfall; Smartestenergy:
   Marubeni). A trustee, nominee or lender holding the shares is never the controller (MUFG Bank and Intesa
   Sanpaolo hold Kilbraur's and Millennium's as security since 2022);
5. the company's parent in NESO's Capacity Market register, resolved the same way; then the last controller's name
   where the chain reaches no known group (`ch_psc`);
6. the company itself.

**Traders by year.** A unit's trader is the lead party of its BM units in the year, from gb-power-data's BM-unit
table: a documented lead party (`bmu_lead_parties.csv`) where one covers the date; else the register snapshot of
the year (Elexon publishes only the register of the day, so snapshots are kept from 5 Jun 2026 on); else today's
register as a proxy, checked against the unit's ownership: where a dated source (a curated row or DESNZ's list)
shows a different operator in the year than at the snapshot, the unit changed hands since and its trader in the
year is its operator then. So Damhead Creek and Rye House are Drax's to trade in 2018-2020 (VPI's today), West
Burton B EDF's until Aug 2021 (EIG's today), and Severn Calon's until 2025 (Centrica bought it in May 2026). A
joint venture's own company in the register (London Array, Greater Gabbard) trades for the venture's operator.
Only BM units active in the year (dated by their output) have a trader.

**Outputs.**

| File | One row per | Columns |
| --- | --- | --- |
| `ownership_data/built/unit_roles_{year}.csv` | stack unit, role, group | year, unit, kind, role, group, share, capacity_mw, company (as registered), basis (curated, alias, name, address, site, cm_parent, self, jv_operator, operator_at_date, register), source (the register record, or how the trader was found), bmu_ids |
| `ownership_data/built/groups_{year}.csv` | group | owner_mw (equity-weighted where shares are known), operator_mw, trader_mw |
| `<scenario>/unit_owners.csv` | scenario unit, role, group | unit, role, group, share; a merged unit's shares are its stack units' shares by capacity; a joint venture whose shares no source gives is split equally among its named owners |

`unit_owners.csv` sits beside the framework's files and the framework does not read it: it is for the tool
(portfolios, the levy, figures) and for analysis. The levy (`egl`) reads it first (`operators_from_view`), so a
scenario folder carries its owners where gb-power-data is not reachable, as on the HPC; only building owners
(`build`, `ownership`) needs it, and `ownership_data/` is not read at run time. The `unit_operator` column of
`powerplant_units_learning.csv` is written by `convert.write_learning` from `owners.portfolio_agents`: the traders
of this table (`owners.bidders`), or the operators with `--agent-level operator`. `agents` rewrites it in a built
folder (`convert.rewrite_learning`), with `unit_operators_learning.csv` and the learning block of
`scenario_meta.json` (which records `agent_level` and `min_portfolio_mw`), and nothing else. Groups are spelled as
gb-power-data's tables spell them (`one_spelling`), so an operator and a trader name one company alike. Without
gb-power-data, `test_ownership.py` runs the tests of the model's own rules and skips the rest.

## 3. What it gives now

Units with an owner of record, 2018-2025: 103 to 147 named units (2023: 27 plants, 58 farms, 37 CfD units),
about half of each scenario's capacity (60% in 2018, 51% in 2023 and 2025); 31 operators in 2018, 39 in 2023, 60
in 2025. Largest operators in 2023: RWE 9.9 GW, SSE 8.1 GW, Uniper 5.5 GW, Orsted 4.1 GW, VPI 2.85 GW, EPUKi 2.7 GW
of the scenario's named units. In 2023, 15 named units are traded by a company other than their operator (8 in
2018, 16 in 2025): eight whose output an offtaker trades (Statkraft: An Suidhe, Carraig Gheal, Coire na Cloiche;
Eneco: both Crystal Rig units; Smartestenergy, Marubeni's: Farr, Tralorg; EDF: Corriemoillie); three whose project
company Companies House places with an owner other than the RO holder (Freasdail: TRIG; Minnygap: Aviva
Investors; Aikengall II: CSM Sustainable Energy); and four whose lead party resolves to itself (Kilbraur and
Millennium Wind Energy, whose only holder of record since Dec 2022 is a lender; OnPath Energy, Banks Renewables
renamed, for Kype Muir and Middle Muir in 2023-2024, when Brookfield's filing gives it 25-50% of the votes only).
Before the Companies House lookup of the lead parties there were 19: Aberdeen Bay, Brockloch Rig, Galawhistle and
A'Chruach are traded by their operators' own project companies.

**Against the tables of 6 Oct 2026, morning** (this folder's own copies, today's lead parties), for every year
2018-2025 on the register stacks (scenario units changed per year, rebuilt on 7 Oct 2026: owners 14-24, operators
4-7, traders 10-13):

- traders are the year's: Damhead Creek, Rye House (Drax) and West Burton B (EDF) in 2018-2020, Severn (Calon)
  in 2018-2025, Robin Rigg and Rampion (E.ON) in 2018; a joint venture's company trades for its operator
  (Dorenell, Fallago Rig: EDF; Gunfleet Sands: Orsted; Moray East: Ocean Winds; Teesside: Macquarie); Clyde has a
  trader (SSE); units not active in the BM in the year have none (Baglan Bay from 2021, its last output in July
  2020; two CfD farms before their first output, Tralorg in 2020-2021 and Solwaybank in 2020-2023), nor have the
  closed coal plants and Deeside, whose BM units are not in the register snapshot of June 2026 (Cottam and
  Eggborough in 2018, Aberthaw B in 2018-2019, Deeside in every year);
- the plants DESNZ names differently or leaves out have their owners: Grain ("Grain CHP") is Uniper's in
  2018-2020, Sutton Bridge Calon's in 2023-2024;
- owners carry the sourced shares of the joint ventures: the CfD farms (Hornsea One and Two, Moray East, Triton
  Knoll, Dudgeon, Beatrice, Walney Extension, Burbo Bank Extension, East Anglia One), Fallago Rig (Federated Hermes
  80, EDF 20), Lincs (Macquarie's 75 before May 2022, later Octopus in), Humber Gateway, Ormonde, Sheringham
  Shoal, Race Bank, Rampion, Galloper, Westermost Rough, Dorenell and Neart na Gaoithe, and Gunfleet Sands (Orsted
  50.1, DBJ 24.95, Marubeni and from 2019 JERA 24.95: its RO holder, Orsted Burbo (UK), also holds Burbo Bank, so
  the venture that trades its BM units gives the owners);
- operators: London Array is run by its own company until RWE took over operations at the start of 2023 (The
  Crown Estate's tables list London Array Ltd as operator at the ends of 2018-2020), not by E.ON and RWE; Dorenell
  by EDF; Teesside by Macquarie; the Banks wind farms group to Banks;
- the battery fleet by trader follows the BM units active in each year (2018: Roosecote, Centrica's, and Redfield
  Road, Conrad's), no longer every battery of today's register.

**The learning agents of the scenario folders** were rewritten on 7 Oct 2026 at trader level (`agents --years
2018 2025`; the folders were built on 5 October with the operators of the tables of then, and nothing else in
them changed). Agents/learning units, 5 Oct -> now: 2018 6/46 -> 13/61, 2019 6/48 -> 12/63, 2020 5/46 -> 12/63,
2021 8/52 -> 10/56, 2022 9/58 -> 11/61, 2023 10/60 -> 11/61, 2024 10/60 -> 11/61, 2025 9/57 -> 11/60. Most of it is
the corrected tables: in 2018-2020 Uniper, InterGen, EPUKi, Calon, Triton Power and EDF are found (the older DUKES
editions name their plants differently), Drax is one group (it was "Drax Power Ltd" and "Drax Power", and missing
in 2020) and E.ON's farms make it an agent in 2018; Calon comes in from 2021, Triton Power from 2022, EPUKi in 2021
and 2025; EDF drops out from 2023 (West Burton B is EIG's since Aug 2021); SSE gains Marchwood. The level itself
moves three wind farms against operators
on today's data: Farr (92 MW, Smartestenergy) and An Suidhe (19 MW, Statkraft) leave RWE's portfolio and no longer
learn, since their traders have no portfolio, and Corriemoillie (48 MW, EDF) joins EDF's in 2018-2020. The gas
portfolios are the same at both levels: every CCGT's lead party is its operator's own trading company. Crystal Rig
(Eneco) learns at neither level. No scenario folder has the `day_ahead_storage_traders` case.

**With the CfD units in the portfolios** (cases `learning_cfd*`, written into every folder on 7 Oct 2026; the
other cases and files unchanged, checked by hash): agents/learning units 2018 14/70, 2019 13/78, 2020 13/81, 2021
11/76, 2022 12/81, 2023 12/82, 2024 13/89, 2025 14/92, the merged CfD units split into 7 (2018) to 52 (2025)
contracts. Orsted gains Hornsea One and Two, Walney Extension and Burbo Bank Extension (2.0 -> 5.5 GW in 2024),
RWE Triton Knoll and three onshore farms, ScottishPower East Anglia One, SSE Beatrice, Vattenfall Solwaybank (from
2024); Equinor becomes an agent in every year (Dudgeon's three contracts with Sheringham Shoal), Ocean Winds in
2024-2025 (Moray East) and EDF in 2025 (Neart na Gaoithe and Dorenell with Fallago Rig and Corriemoillie).

## 4. Known limits

- **Traders before June 2026 are a proxy.** Elexon publishes only today's register; for earlier years today's
  lead party stands in, corrected only where a dated source shows a change of hands. Offtake agreements that
  changed without a sale (a wind farm moving from one offtaker to another) are not seen, so today's offtaker bids
  in every year (Eneco for Crystal Rig, EDF for Corriemoillie). Keeping a register snapshot every month builds the
  history from now on.
- **A lead party that resolves to itself is taken as the unit's own project company,** so the unit's operator
  bids (Kilbraur and Millennium in 2022-2025). A third-party trader that no rule, source or Companies House parent
  places would be taken the same way; on 7 Oct 2026 no named plant or farm has one.
- **The trader is the group of the BM lead party.** A trading arm is placed in its parent's group (Smartestenergy
  in Marubeni's), and a joint venture's partners that take their shares of the output to trade themselves are not
  seen: the BM has one lead party per unit, and a venture's own company resolves to its operator.
- **Activity is dated by output.** gb-power-data counts a BM unit as active in a year if it produced in it. The
  units without files had their notifications pulled on 7 Oct 2026; those that notified nothing before 2026 (Dogger
  Bank B's 1.2 GW among them) count in no year, and three small units that submit none (55 MW) count in every
  year.
- **The CM register's parent is today's.** NESO updates it after a sale. It resolves companies to groups, so a
  company sold since takes its buyer's group in every year (Flexitricity: Drax), unless a dated row says
  otherwise.
- **DESNZ lags sales, lists stations at the end of May and only major power producers.** DUKES 2026 still has
  Saltend under Energy Capital Partners and Severn under Calon; dated rows in `company_owners.csv` correct what is
  known. A plant that closed between January and May is read from the edition before.
- **A few joint-venture shares are open or partly verified**: see gb-power-data's `docs/ownership_references.md`
  (status per row). Where no source gives shares the scenario view splits the unit equally, and where the known
  shares do not add up to one the rest is `unresolved`.
- **Half the capacity has no owner of record.** The model's merged classes (RO solar and small wind by GSP group
  and band, the CfD solar classes, FiT, embedded generation) stand for many small plants; their owners would
  need the plants behind each class (REPD, the RO and CfD registers) and a share per class.
- **The model's names change.** `plants.csv` maps the model's plant names to gb-power-data's stations; a new plant
  name in the merit-order model needs a row (a farm unit finds its station by name).
- **The battery fleets differ.** The model's fleet (4.2 GW in 2025) is not the BM register's (5.2 GW active) nor
  the CM register's (7.9 GW in force); only the shares are carried over.
- **The model keeps two closed CCGTs.** Baglan Bay (`Bage`, 552 MW) is offered in every half-hour of 2021-2025
  (and clears in 14-29% of them in 2022-2025), though its BM units last produced in July 2020 and DESNZ drops it
  from 2023: it has no trader from 2021 and no owner from 2023. Deeside (515 MW), mothballed for synchronous
  compensation, has no BM unit in the 2026 register and no DUKES entry, yet is offered in every half-hour of
  2019-2025; it has no trader, and with Saltend it makes Triton Power a learning portfolio. Retiring both is the
  merit-order model's register-fleet build to do.

## 5. Data that could be added

Ordered by what it adds for the agents, with the effort as estimated on 6 Oct 2026. Rows marked *done* are in
gb-power-data.

| Data | What it adds | Value for the ABM | Access | Effort |
| --- | --- | --- | --- | --- |
| Companies House, persons with significant control (PSC) | the corporate owners of a company with 25% or more, in bands, by company number; resolves project companies to parents, and dated changes of control | turns "self" project companies into groups and dates sales; checks joint ventures | the API with a free key; the CfD register already carries company numbers | *done* in gb-power-data (`pull.companies_house_pull`, `build.build_company_control`, basis `ch_psc`), pulled on 7 Oct 2026: 183 CfD project companies and 50 BM lead parties that resolved to themselves (numbers by Companies House's company search on Elexon's spelling), with their parents two levels up; trustees, nominees and lenders are never controllers |
| The Crown Estate's offshore wind ownership tables | owners with shares and the operator of every offshore farm at the end of 2018, 2019, 2020, 2023 and 2024, and the transactions of 2019-2022 | the offshore joint ventures by year | report PDFs and the live page (own use only) | *done*: read into `company_owners.csv`, and gb-power-data's `build.build_crown_estate --strict` checks it against every table (171 farm-dates, 13 differences, all explained) |
| Lead-party history | the BM register pulled monthly, kept as dated snapshots | traders by year instead of a proxy | Elexon Insights API (free) | *done* for the build (snapshots by date) and gb-power-data's `scripts/snapshot_registers.py`, run on the 1st of every month by a scheduled task since 7 Oct 2026 (first run 1 Nov 2026) |
| NESO response and reserve auction results by unit (Dynamic Containment, Regulation, Moderation, Balancing Reserve, Quick and Slow Reserve) | the participant that sells each unit's services, per auction | who trades batteries' ancillary services; revenue stacking against energy arbitrage; the batteries' outside option | NESO data portal (free; daily files by unit since May 2026) | 1-2 days |
| BM acceptances (BOALF) and settlement cashflows per BM unit (Elexon P114, S0142) | each company's accepted volume and payments per half-hour, by direction and zone | the market structure of the BM; the BM as an outside option in venue choice; company-level validation | Elexon Insights (BOALF, free); Elexon Portal (P114, free registration) | 2-3 days for a full pull |
| Suppliers' market shares | the retail side of integrated companies (Ofgem retail market indicators) | net position: an integrated company short on generation gains less from a high price | Ofgem data portal | 1 day |
| NESO TEC register | transmission connections by customer and status | owners of the transmission pipeline (offshore wind, batteries) for 2030 | NESO data portal (free) | 1 day |
| REMIT outage messages (Elexon IRIS / BMRS REMIT) | unavailability per asset with the market participant's ACER code | availability by company (withholding tests); a second link from participants to assets | Elexon (free) | 2 days |
| PPA offtakers | who buys an RO wind farm's output | who bears the price risk of RO wind; part of the gap between owner and trader | no public register; BM lead parties and press releases in part | open-ended |
| Ofgem's generation market shares by company | annual shares and concentration index | a check on the owners' totals | Ofgem wholesale market indicators | small |
| Battery databases (Modo, Solar Media, Cornwall Insight) | owner, optimiser and contract per battery | validation of the trader/owner split | commercial | n/a |

What exchange data cannot give: the day-ahead and intraday books are anonymous, and the exchanges publish at
most who may trade (member lists), not who traded what; Ofgem holds trade-level REMIT data, which is not public.
Company structure in those markets can only be inferred from the supply side (who owns and trades the capacity
that sets prices) or from the BM, where every action has its unit.

## 6. Next steps

1. In gb-power-data: a first Companies House pull (a free key). Done on 7 Oct 2026, with the BM lead parties that
   resolved to themselves; the monthly register snapshot is scheduled, and the units without activity files that
   notify had their notifications pulled.
2. The concentration figures by market (the agreed order: Capacity Market register, the linking table, the
   concentration by register and market, BM payments by company, the pivotal tests in the day-ahead market),
   counted by trader for the ability to move the price and by owner for whose profit it is.
3. Done on 7 Oct 2026: the learning agents of the folders are the traders (section 3). The policies in
   `gb_2023/learned_strategies` were trained on the agents of 4-5 Oct and need training again; the HPC's copies
   of the folders need the new `powerplant_units_learning.csv`, `unit_operators_learning.csv` and
   `scenario_meta.json` (and `unit_owners.csv`).

The superseded copies of the curated tables (`aliases.csv`, `company_owners.csv` of 6 Oct 2026, morning, and
`plants.csv` with its site patterns) are kept in `ownership_data/superseded_2026-10-06/` for comparison; nothing
reads them.
