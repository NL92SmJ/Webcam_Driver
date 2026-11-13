import glob
import os
import matplotlib.pyplot as plt
import pandas as pd

# 1. Neueste CSV-Logdatei finden
csv_files = glob.glob("dms_perf_log_*.csv")
if not csv_files:
  csv_files = glob.glob("dms_log_*.csv")

if not csv_files:
  print("Keine CSV-Logdatei gefunden!")
  exit()

latest_file = max(csv_files, key=os.path.getctime)
print(f"Lese Logdatei ein: {latest_file}")

# 2. Daten einlesen
df = pd.read_csv(latest_file)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df_valid = df[df["Head_Pose_State"] != "Kein Gesicht"].copy()

if len(df_valid) == 0:
  print("Die Logdatei enthält keine gültigen Gesichtsdaten.")
  exit()

# 3. Matplotlib Dashboard erstellen (3 Zeitreihen-Diagramme untereinander)
fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
fig.suptitle(
    f"DMS Telemetrie & Performance Dashboard\nDatei: {latest_file}",
    fontsize=14,
    fontweight="bold",
)

# --- Plot 1: Kopfhaltung (Yaw, Pitch, Roll) ---
axes[0].plot(
    df_valid["Timestamp"],
    df_valid["Yaw"],
    label="Yaw (Links/Rechts)",
    color="blue",
    linewidth=1.5,
)
axes[0].plot(
    df_valid["Timestamp"],
    df_valid["Pitch"],
    label="Pitch (Oben/Unten)",
    color="orange",
    linewidth=1.5,
)
axes[0].set_ylabel("Winkel (Grad)")
axes[0].set_title("Kopf-Orientierung im 3D-Raum")
axes[0].legend(loc="upper right")
axes[0].grid(True, alpha=0.3)

# --- Plot 2: Augen (EAR) & Mund (MAR) ---
axes[1].plot(
    df_valid["Timestamp"],
    df_valid["EAR"],
    label="EAR (Augen)",
    color="green",
    linewidth=1.5,
)
axes[1].axhline(0.22, color="red", linestyle="--", label="Blinzel-Schwelle")
axes[1].plot(
    df_valid["Timestamp"],
    df_valid["MAR"],
    label="MAR (Mund)",
    color="purple",
    linewidth=1.5,
)
axes[1].axhline(0.45, color="orange", linestyle="--", label="Gähn-Schwelle")
axes[1].set_ylabel("Ratio")
axes[1].set_title("Müdigkeits-Metriken (EAR & MAR)")
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)

# --- Plot 3: Hardware-Auslastung (CPU & RAM) ---
if "CPU_Percent" in df_valid.columns and "RAM_Percent" in df_valid.columns:
  axes[2].plot(
      df_valid["Timestamp"],
      df_valid["CPU_Percent"],
      label="CPU-Last (Skript) [%]",
      color="crimson",
      linewidth=1.5,
  )
  axes[2].plot(
      df_valid["Timestamp"],
      df_valid["RAM_Percent"],
      label="RAM-Last (System) [%]",
      color="dodgerblue",
      linewidth=1.5,
  )
  axes[2].set_ylabel("Auslastung (%)")
  axes[2].set_xlabel("Zeitverlauf")
  axes[2].set_title("Hardware-Performance (CPU & RAM)")
  axes[2].legend(loc="upper right")
  axes[2].grid(True, alpha=0.3)
else:
  axes[2].text(
      0.5,
      0.5,
      "Keine CPU/RAM-Daten in dieser CSV vorhanden",
      horizontalalignment="center",
      verticalalignment="center",
      transform=axes[2].transAxes,
  )

plt.tight_layout()

# Zeitreihen-Dashboard speichern
dashboard_filename = latest_file.replace(".csv", "_dashboard.png")
fig.savefig(dashboard_filename, dpi=300)
print(f"Gesamt-Dashboard erfolgreich gespeichert unter: {dashboard_filename}")

# --- SEPARATES DIAGRAMM: Häufigkeit der Kopfhaltungen (Balken) ---
plt.figure(figsize=(8, 5))
pose_counts = df_valid["Head_Pose_State"].value_counts()
bars = plt.bar(
    pose_counts.index,
    pose_counts.values,
    color=["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"],
    edgecolor="black",
)
plt.ylabel("Anzahl Frames")
plt.title(
    "Häufigkeit der Kopfhaltungen", fontsize=12, fontweight="bold"
)
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
bar_filename = latest_file.replace(".csv", "_barchart.png")
plt.savefig(bar_filename, dpi=300)
print(f"Balkendiagramm gespeichert unter: {bar_filename}")

# --- SEPARATES DIAGRAMM: Tortendiagramm (Prozent) ---
plt.figure(figsize=(6, 6))
plt.pie(
    pose_counts,
    labels=pose_counts.index,
    autopct="%1.1f%%",
    startangle=140,
    colors=["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"],
    wedgeprops={"edgecolor": "black", "linewidth": 1},
)
plt.title(
    "Verteilung der Kopfhaltungen (%)", fontsize=12, fontweight="bold"
)
plt.tight_layout()
pie_filename = latest_file.replace(".csv", "_distribution.png")
plt.savefig(pie_filename, dpi=300)
print(f"Tortendiagramm gespeichert unter: {pie_filename}")

# Alle Diagramme anzeigen
plt.show()