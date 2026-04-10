import pandas as pd
import numpy as np

# BN bookstore problem
# ------------------------------------------------------------
# using the demand data to:
# (1) find the order quantity that maximizes BN’s average profit
# (2) see if changing the contract (wholesale / buy-back) can make
#     both BN and the publisher better off
#
# assumptions:
# - BN sells books at $24
# - BN pays publisher $12 per book (original contract)
# - publisher cost is $1 per book
# - leftover books are salvaged for $3 if no buy-back
#
# if we add a buy-back:
# - BN returns leftovers to publisher for price b
# - publisher then salvages those for $3
# ------------------------------------------------------------


# ----------------------------
# load demand data
# ----------------------------
file_path = "/mnt/data/MGT155_BookDemand_data(1).csv"
df = pd.read_csv(file_path)
demand = df["Demand"].dropna().astype(int).to_numpy()

# only need to check order quantities that appear in the data
# (profit only changes at those points)
Q_candidates = np.unique(np.concatenate(([0], demand)))


# ----------------------------
# parameters
# ----------------------------
retail_price = 24
wholesale_price = 12
publisher_unit_cost = 1
salvage_value = 3


# ----------------------------
# profit functions
# ----------------------------
def baseline_metrics(Q):
    # original contract (no buy-back)

    sales = np.minimum(demand, Q)
    leftover = np.maximum(Q - demand, 0)

    # BN: revenue from sales + salvage - wholesale cost
    bn_profit = (retail_price * sales + salvage_value * leftover - wholesale_price * Q).mean()

    # publisher: earns margin on every unit sold to BN
    publisher_profit = (wholesale_price - publisher_unit_cost) * Q

    total_profit = bn_profit + publisher_profit

    return {
        "Q": int(Q),
        "bn_profit": float(bn_profit),
        "publisher_profit": float(publisher_profit),
        "total_profit": float(total_profit),
    }


def buyback_metrics(Q, w, b):
    # contract with buy-back

    sales = np.minimum(demand, Q)
    leftover = np.maximum(Q - demand, 0)

    # BN: sells books + gets buy-back on leftovers
    bn_profit = (retail_price * sales + b * leftover - w * Q).mean()

    # publisher: earns margin on sales + salvages returned books
    publisher_profit = ((w - publisher_unit_cost) * Q + (salvage_value - b) * leftover).mean()

    total_profit = bn_profit + publisher_profit

    return {
        "Q": int(Q),
        "w": float(w),
        "b": float(b),
        "bn_profit": float(bn_profit),
        "publisher_profit": float(publisher_profit),
        "total_profit": float(total_profit),
    }


# ----------------------------
# QUESTION 1
# ----------------------------
# find Q that maximizes BN profit under original contract
q1_results = [baseline_metrics(Q) for Q in Q_candidates]
q1_best = max(q1_results, key=lambda x: x["bn_profit"])

print("QUESTION 1")
print(f"Optimal order quantity: {q1_best['Q']:,}")
print(f"BN average profit: ${q1_best['bn_profit']:,.2f}")
print(f"Publisher average profit: ${q1_best['publisher_profit']:,.2f}")
print()


# ----------------------------
# QUESTION 2
# ----------------------------
# try different wholesale + buy-back prices
# goal: find a contract where BOTH BN and publisher make more than Q1

pareto_contracts = []

for w in np.arange(8, 14.5, 0.5):
    for b in np.arange(3, 12.5, 0.5):

        # BN chooses Q that maximizes its own profit under this contract
        results = [buyback_metrics(Q, w, b) for Q in Q_candidates]
        best_for_bn = max(results, key=lambda x: x["bn_profit"])

        # check if both are better than Q1
        if (
            best_for_bn["bn_profit"] > q1_best["bn_profit"] and
            best_for_bn["publisher_profit"] > q1_best["publisher_profit"]
        ):
            pareto_contracts.append(best_for_bn)


if pareto_contracts:
    # pick the one that gives highest total profit
    q2_best = max(pareto_contracts, key=lambda x: x["total_profit"])

    print("QUESTION 2")
    print("found a contract that improves both sides")
    print(f"wholesale price: ${q2_best['w']:.2f}")
    print(f"buy-back price: ${q2_best['b']:.2f}")
    print(f"new order quantity: {q2_best['Q']:,}")
    print(f"BN profit: ${q2_best['bn_profit']:,.2f}")
    print(f"Publisher profit: ${q2_best['publisher_profit']:,.2f}")
else:
    print("QUESTION 2")
    print("no better contract found in this range")


# ----------------------------
# benchmark: fully coordinated supply chain
# ----------------------------
# this is the “ideal” case where we maximize total profit

def integrated_profit(Q):
    sales = np.minimum(demand, Q)
    leftover = np.maximum(Q - demand, 0)
    return (retail_price * sales + salvage_value * leftover - publisher_unit_cost * Q).mean()


best_integrated_Q = max(Q_candidates, key=integrated_profit)

print()
print("Supply chain benchmark")
print(f"best total-profit order quantity: {best_integrated_Q:,}")
print(f"max total average profit: ${integrated_profit(best_integrated_Q):,.2f}")
