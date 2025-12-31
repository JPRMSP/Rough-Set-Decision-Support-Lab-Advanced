import streamlit as st
import itertools
import pandas as pd

st.set_page_config(page_title="Rough Set Decision Support Lab", layout="centered")

st.title("🧠 Rough Set Decision Support Lab — Advanced")
st.caption("Pure Rough Set Theory | No datasets | No ML models")

st.markdown("""
Enter a **decision table** manually.

Each row = object  
Each column = attribute  
Final column = **Decision**
""")

# -------- INPUTS --------
rows = st.number_input("Number of rows (objects)", 2, 20, 6)
cols = st.number_input("Number of condition attributes", 1, 6, 3)

attr_names = [f"A{i+1}" for i in range(cols)]
columns = attr_names + ["Decision"]

st.subheader("📋 Fill Decision Table")
data = []

for r in range(int(rows)):
    row = []
    for c in columns:
        row.append(st.text_input(f"{c} (row {r+1})", key=f"{c}_{r}"))
    data.append(row)

df = pd.DataFrame(data, columns=columns)
st.dataframe(df, use_container_width=True)


# -------- CORE FUNCTIONS --------
def equivalence_classes(df, attributes):
    groups = {}
    for idx, row in df.iterrows():
        key = tuple(row[a] for a in attributes)
        groups.setdefault(key, []).append(idx)
    return list(groups.values())

def lower_upper(df, attrs, decision_value):
    eq = equivalence_classes(df, attrs)
    lower, upper = [], []

    for cls in eq:
        decs = set(df.loc[cls]["Decision"])
        if decs == {decision_value}:
            lower.extend(cls)
            upper.extend(cls)
        elif decision_value in decs:
            upper.extend(cls)

    return sorted(set(lower)), sorted(set(upper))

def reducts(df, attrs):
    red = []
    full = equivalence_classes(df, attrs)

    for r in range(1, len(attrs)+1):
        for combo in itertools.combinations(attrs, r):
            if equivalence_classes(df, combo) == full:
                red.append(combo)

    return red

def extract_rules(df, attrs):
    rules = []
    for decision in df["Decision"].unique():
        lower, _ = lower_upper(df, attrs, decision)

        for idx in lower:
            cond = [f"{a}={df.loc[idx, a]}" for a in attrs]
            rule = f"IF " + " AND ".join(cond) + f" THEN Decision = {decision}"
            rules.append(rule)

    return list(sorted(set(rules)))

def inconsistent_objects(df, attrs):
    conflicts = []
    eq = equivalence_classes(df, attrs)

    for cls in eq:
        decs = set(df.loc[cls]["Decision"])
        if len(decs) > 1:
            conflicts.append((cls, list(decs)))

    return conflicts


# -------- RUN --------
if st.button("Run Rough Set Analysis"):
    st.header("📌 Results")

    # VALIDATE
    if df.isna().any().any() or (df == "").any().any():
        st.error("Please fill all cells — empty values break equivalence classes.")
        st.stop()

    decisions = df["Decision"].unique()

    # --- APPROXIMATIONS ---
    st.subheader("1️⃣ Approximations")
    for d in decisions:
        st.markdown(f"### Decision: **{d}**")

        lower, upper = lower_upper(df, attr_names, d)
        boundary = sorted(set(upper) - set(lower))

        st.write("✔ Lower approximation (certain):", lower)
        st.write("✔ Upper approximation (possible):", upper)
        st.write("⚠ Boundary:", boundary)

        accuracy = len(lower) / len(upper) if len(upper) else 0
        st.write("🎯 Accuracy:", round(accuracy, 3))

        st.info("Interpretation: Lower = guaranteed. Upper = possible. Boundary = uncertainty region.")

    # --- REDUCTS ---
    st.subheader("2️⃣ Attribute Reducts")
    rds = reducts(df, attr_names)

    if rds:
        for r in rds:
            st.write("•", list(r))
        st.info("A reduct is the minimal attribute set preserving classification power.")
    else:
        st.warning("No reducts found — table may already be minimal.")

    # --- RULES ---
    st.subheader("3️⃣ Decision Rules (from Lower Approximation)")
    rules = extract_rules(df, attr_names)

    if rules:
        for r in rules:
            st.code(r)
    else:
        st.warning("No certain rules could be generated (too much uncertainty).")

    # --- CONFLICT ANALYSIS ---
    st.subheader("4️⃣ Conflict Analysis")
    conflicts = inconsistent_objects(df, attr_names)

    if conflicts:
        st.error("Conflicts detected:")
        for group, values in conflicts:
            st.write(f"Objects {group} → conflicting decisions {values}")
        st.caption("This occurs when identical conditions lead to different decisions.")
    else:
        st.success("No conflicts — knowledge base is consistent.")

    st.success("Analysis completed — explain each section during presentation to impress your professor!")
