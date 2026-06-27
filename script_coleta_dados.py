import re
import pandas as pd

log_file = "runtime.log"

# -------------------------
# Expressões regulares
# -------------------------

timestamp = r"\(\d+\)\((?P<time>[\d\-]+\s[\d:,]+)\)"

create_re = re.compile(
    timestamp +
    r".*@locatableAction.*Task (?P<task>\d+), CE name (?P<func>[\w\.]+)"
)

assign_re = re.compile(
    timestamp +
    r".*@gnWorkerAndImpl.*Task (?P<task>\d+).*worker (?P<worker>\S+)"
)

running_re = re.compile(
    timestamp +
    r".*@actionRunning.*Task (?P<task>\d+)"
)

completed_re = re.compile(
    timestamp +
    r".*@actionCompleted.*Task (?P<task>\d+)"
)

tasks = {}

# -------------------------
# Leitura do log
# -------------------------

with open(log_file) as f:

    for line in f:

        m = create_re.search(line)
        if m:
            tid = int(m.group("task"))
            tasks.setdefault(tid, {})
            tasks[tid]["task_id"] = tid
            tasks[tid]["task_func_name"] = m.group("func")
            tasks[tid]["create_time"] = pd.to_datetime(
                m.group("time"),
                format="%Y-%m-%d %H:%M:%S,%f"
            )
            continue

        m = assign_re.search(line)
        if m:
            tid = int(m.group("task"))
            tasks.setdefault(tid, {})
            tasks[tid]["worker"] = m.group("worker")
            tasks[tid]["assign_time"] = pd.to_datetime(
                m.group("time"),
                format="%Y-%m-%d %H:%M:%S,%f"
            )
            continue

        m = running_re.search(line)
        if m:
            tid = int(m.group("task"))
            tasks.setdefault(tid, {})
            tasks[tid]["start_time"] = pd.to_datetime(
                m.group("time"),
                format="%Y-%m-%d %H:%M:%S,%f"
            )
            continue

        m = completed_re.search(line)
        if m:
            tid = int(m.group("task"))
            tasks.setdefault(tid, {})
            tasks[tid]["end_time"] = pd.to_datetime(
                m.group("time"),
                format="%Y-%m-%d %H:%M:%S,%f"
            )

# -------------------------
# DataFrame
# -------------------------

df = pd.DataFrame(tasks.values())

print(df.head())

# Apenas tasks completas
df = df.dropna(subset=["start_time", "end_time"])

# Tempo de execução
df["tempo_execucao_s"] = (
    df["end_time"] -
    df["start_time"]
).dt.total_seconds()

# Tempo na fila (opcional)
if "create_time" in df.columns:
    df["tempo_fila_s"] = (
        df["start_time"] -
        df["create_time"]
    ).dt.total_seconds()

# -------------------------
# Resumo
# -------------------------

resumo = (
    df
    .groupby("task_func_name")
    .agg(
        quantidade=("task_id", "count"),
        media=("tempo_execucao_s", "mean"),
        mediana=("tempo_execucao_s", "median"),
        minimo=("tempo_execucao_s", "min"),
        maximo=("tempo_execucao_s", "max"),
        desvio=("tempo_execucao_s", "std"),
        total=("tempo_execucao_s", "sum")
    )
    .reset_index()
)

resumo["coef_var_%"] = (
    resumo["desvio"] /
    resumo["media"] * 100
)

print("\n===== RESUMO =====")
print(resumo.round(3))

print("\n===== PRIMEIRAS TASKS =====")
print(
    df[
        [
            "task_id",
            "task_func_name",
            "worker",
            "tempo_execucao_s",
            "tempo_fila_s"
        ]
    ].head(20)
)
