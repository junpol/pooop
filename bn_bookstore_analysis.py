import pandas as pd
import numpy as np

# ------------------------------------------------------------
# BN bookstore exercise
# ------------------------------------------------------------
# This script answers two questions using the uploaded demand data:
# 1) Under the original contract, what order quantity maximizes BN's
#    average profit, and what are the resulting average profits for
#    BN and the publisher?
# 2) Can a new contract (wholesale price and/or buy-back price) make
#    both BN and the publisher better off than in Question 1?
#
# Assumptions used (from the standard BN/publisher newsvendor exercise):
# - Retail price charged by BN to customers: $24
# - Original wholesale price paid by BN to publisher: $12
# - Publisher marginal production cost: $1
# - If there is no buy-back contract, BN discounts unsold books and
#   earns a salvage value of $3 per leftover book.
#
# If a buy-back contract is introduced, we assume:
# - BN returns leftover books to the publisher for buy-back price b
# - The publisher then recovers the same salvage value of $3 per
#   returned leftover book
# ------------------------------------------------------------

# ----------------------------
# Load demand data
# ----------------------------
file_path = "/mnt/data/MGT155_BookDemand_data(1).csv"
df = pd.read_csv(file_path)
demand = df["Demand"].dropna().astype(int).to_numpy()

# Candidate order quantities:
# In a newsvendor problem with empirical demand data, the objective only
# changes slope at observed demand values, so checking 0 and all observed
# demand values is enough.
Q_candidates = np.unique(np.concatenate(([0], demand)))

# ----------------------------
# Model parameters
# ----------------------------
retail_price = 24
wholesale_price = 12
publisher_unit_cost = 1
salvage_value = 3


# ----------------------------
# Profit functions
# ----------------------------
def baseline_metrics(Q, p=retail_price, w=wholesale_price,
                     c=publisher_unit_cost, s=salvage_value):
    """
    Original contract (no buy-back):
    - BN sells at price p
    - BN pays wholesale price w for every unit ordered
    - BN salvages leftover books for s each
    - Publisher earns w - c on every unit ordered
    """
    sales = np.minimum(demand, Q)
    leftover = np.maximum(Q - demand, 0)

    bn_profit = (p * sales + s * leftover - w * Q).mean()
    publisher_profit = (w - c) * Q
    total_profit = bn_profit + publisher_profit

    return {
        "Q": int(Q),
        "bn_profit": float(bn_profit),
        "publisher_profit": float(publisher_profit),
        "total_profit": float(total_profit),
        "avg_sales": float(sales.mean()),
        "avg_leftover": float(leftover.mean())
    }


def buyback_metrics(Q, w, b, p=retail_price,
                    c=publisher_unit_cost, s=salvage_value):
    """
    Buy-back contract:
    - BN still sells at price p
    - BN pays wholesale price w
    - BN returns leftover books to publisher for buy-back price b
    - Publisher recovers salvage value s on returned leftovers
    """
    sales = np.minimum(demand, Q)
    leftover = np.maximum(Q - demand, 0)

    bn_profit = (p * sales + b * leftover - w * Q).mean()
    publisher_profit = ((w - c) * Q + (s - b) * leftover).mean()
    total_profit = bn_profit + publisher_profit

    return {
        "Q": int(Q),
        "w": float(w),
        "b": float(b),
        "bn_profit": float(bn_profit),
        "publisher_profit": float(publisher_profit),
        "total_profit": float(total_profit),
        "avg_sales": float(sales.mean()),
        "avg_leftover": float(leftover.mean())
    }


# ----------------------------
# Question 1: best BN order quantity
# ----------------------------
q1_results = [baseline_metrics(Q) for Q in Q_candidates]
q1_best = max(q1_results, key=lambda x: x["bn_profit"])

print("QUESTION 1")
print(f"Optimal BN order quantity: {q1_best['Q']:,}")
print(f"BN average profit: ${q1_best['bn_profit']:,.2f}")
print(f"Publisher average profit: ${q1_best['publisher_profit']:,.2f}")
print(f"Total supply chain average profit: ${q1_best['total_profit']:,.2f}")
print()


# ----------------------------
# Question 2: search for an improved contract
# ----------------------------
# We search a grid of wholesale prices and buy-back prices.
# The goal is to find a contract where BOTH BN and the publisher earn
# more average profit than in Question 1.
#
# Among all Pareto-improving contracts found, we choose the one with the
# highest total supply chain average profit.
# ----------------------------
pareto_improving_contracts = []

# Search grid (0.5 increments keep the search readable and flexible)
for w in np.arange(8, 14.5, 0.5):
    for b in np.arange(3, 12.5, 0.5):
        # BN chooses Q to maximize its own average profit under this contract.
        contract_results = [buyback_metrics(Q, w, b) for Q in Q_candidates]
        best_for_bn = max(contract_results, key=lambda x: x["bn_profit"])

        if (
            best_for_bn["bn_profit"] > q1_best["bn_profit"] and
            best_for_bn["publisher_profit"] > q1_best["publisher_profit"]
        ):
            pareto_improving_contracts.append(best_for_bn)

if pareto_improving_contracts:
    q2_best = max(pareto_improving_contracts, key=lambda x: x["total_profit"])

    print("QUESTION 2")
    print("A Pareto-improving contract exists.")
    print(f"Chosen wholesale price: ${q2_best['w']:.2f}")
    print(f"Chosen buy-back price: ${q2_best['b']:.2f}")
    print(f"BN order quantity under new contract: {q2_best['Q']:,}")
    print(f"Improved BN average profit: ${q2_best['bn_profit']:,.2f}")
    print(f"Improved publisher average profit: ${q2_best['publisher_profit']:,.2f}")
    print(f"Total supply chain average profit: ${q2_best['total_profit']:,.2f}")
else:
    q2_best = None
    print("QUESTION 2")
    print("No Pareto-improving contract was found in the search grid.")


# ----------------------------
# Optional: show the integrated supply chain optimum
# ----------------------------
# This is the order quantity that maximizes total supply chain profit,
# not BN's standalone profit.
# ----------------------------
def integrated_profit(Q, p=retail_price, c=publisher_unit_cost, s=salvage_value):
    sales = np.minimum(demand, Q)
    leftover = np.maximum(Q - demand, 0)
    return (p * sales + s * leftover - c * Q).mean()

best_integrated_Q = max(Q_candidates, key=integrated_profit)
print()
print("Integrated supply chain benchmark")
print(f"Profit-maximizing total-chain order quantity: {best_integrated_Q:,}")
print(f"Maximum total supply chain average profit: ${integrated_profit(best_integrated_Q):,.2f}")
