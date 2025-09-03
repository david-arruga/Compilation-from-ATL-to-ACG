from acg import CGS

cgs1 = CGS()

cgs1.add_proposition("safe")
cgs1.add_proposition("start")
cgs1.add_proposition("operational")
cgs1.add_proposition("efficient")
cgs1.add_proposition("underpowered")
cgs1.add_proposition("danger")
cgs1.add_proposition("emergency")

cgs1.add_agent("Reactor")
cgs1.add_agent("Valve")

cgs1.add_decisions("Reactor", {"heat", "cool"})
cgs1.add_decisions("Valve", {"open", "lock"})

cgs1.add_state("s0")
cgs1.add_state("s1")
cgs1.add_state("s2")
cgs1.add_state("s3")
cgs1.add_state("s4")

cgs1.set_initial_state("s0")

cgs1.label_state("s0", {"safe","start"})
cgs1.label_state("s1", {"safe","operational","efficient"})
cgs1.label_state("s2", {"safe","operational","underpowered"})
cgs1.label_state("s3", {"operational","danger"})
cgs1.label_state("s4", {"emergency"})

cgs1.add_transition("s0", {("Reactor", "heat"), ("Valve", "open")}, "s0")
cgs1.add_transition("s0", {("Reactor", "heat"), ("Valve", "lock")}, "s1")
cgs1.add_transition("s0", {("Reactor", "cool"), ("Valve", "open")}, "s0")
cgs1.add_transition("s0", {("Reactor", "cool"), ("Valve", "lock")}, "s0")

cgs1.add_transition("s1", {("Reactor", "heat"), ("Valve", "open")}, "s1")
cgs1.add_transition("s1", {("Reactor", "heat"), ("Valve", "lock")}, "s3")
cgs1.add_transition("s1", {("Reactor", "cool"), ("Valve", "open")}, "s2")
cgs1.add_transition("s1", {("Reactor", "cool"), ("Valve", "lock")}, "s2")

cgs1.add_transition("s2", {("Reactor", "heat"), ("Valve", "open")}, "s2")
cgs1.add_transition("s2", {("Reactor", "heat"), ("Valve", "lock")}, "s1")
cgs1.add_transition("s2", {("Reactor", "cool"), ("Valve", "open")}, "s0")
cgs1.add_transition("s2", {("Reactor", "cool"), ("Valve", "lock")}, "s0")

cgs1.add_transition("s3", {("Reactor", "heat"), ("Valve", "open")}, "s3")
cgs1.add_transition("s3", {("Reactor", "heat"), ("Valve", "lock")}, "s4")
cgs1.add_transition("s3", {("Reactor", "cool"), ("Valve", "open")}, "s2")
cgs1.add_transition("s3", {("Reactor", "cool"), ("Valve", "lock")}, "s1")

cgs1.add_transition("s4", {("Reactor", "heat"), ("Valve", "open")}, "s0")
cgs1.add_transition("s4", {("Reactor", "heat"), ("Valve", "lock")}, "s0")
cgs1.add_transition("s4", {("Reactor", "cool"), ("Valve", "open")}, "s0")
cgs1.add_transition("s4", {("Reactor", "cool"), ("Valve", "lock")}, "s0")


cgs2 = CGS()

for p in (
    "cars_go", "cars_wait", "cross", "dont_cross",
    "clear", "busy", "yellow_phase", "violation",
    "crash", "night_mode", "emergency", "sensor_fault"
):
    cgs2.add_proposition(p)

cgs2.add_agent("CarLight")
cgs2.add_agent("PedLight")

cgs2.add_decisions("CarLight", {"green", "yellow", "red"})
cgs2.add_decisions("PedLight", {"wait", "walk"})

for st in ("s0", "s1", "s2", "s3", "s4", "s5"):
    cgs2.add_state(st)

cgs2.set_initial_state("s1")

cgs2.label_state("s0", {"cars_wait", "peds_cross", "busy"})
cgs2.label_state("s1", {"cars_wait", "peds_wait", "clear", "night_mode"})
cgs2.label_state("s2", {"cars_wait", "peds_wait", "yellow_phase", "busy"})
cgs2.label_state("s3", {"cars_go", "peds_wait", "busy"})
cgs2.label_state("s4", {"cars_go", "peds_cross", "violation", "busy"})
cgs2.label_state("s5", {"crash", "emergency_peds", "sensor_fault"})

cgs2.add_transition("s0", {("CarLight", "red"), ("PedLight", "walk")}, "s0")
cgs2.add_transition("s0", {("CarLight", "red"), ("PedLight", "dontWalk")}, "s1")
cgs2.add_transition("s0", {("CarLight", "yellow"), ("PedLight", "walk")}, "s4")
cgs2.add_transition("s0", {("CarLight", "yellow"), ("PedLight", "dontWalk")}, "s2")
cgs2.add_transition("s0", {("CarLight", "green"), ("PedLight", "walk")}, "s4")
cgs2.add_transition("s0", {("CarLight", "green"), ("PedLight", "dontWalk")}, "s3")

cgs2.add_transition("s1", {("CarLight", "red"),   ("PedLight", "walk")}, "s0")
cgs2.add_transition("s1", {("CarLight", "red"),   ("PedLight", "dontWalk")},  "s1")
cgs2.add_transition("s1", {("CarLight", "yellow"),("PedLight", "walk")},      "s4")
cgs2.add_transition("s1", {("CarLight", "yellow"),("PedLight", "dontWalk")},  "s2")
cgs2.add_transition("s1", {("CarLight", "green"), ("PedLight", "walk")},      "s4")
cgs2.add_transition("s1", {("CarLight", "green"), ("PedLight", "dontWalk")},  "s3")

cgs2.add_transition("s2", {("CarLight", "red"),   ("PedLight", "walk")},      "s0")
cgs2.add_transition("s2", {("CarLight", "red"),   ("PedLight", "dontWalk")},  "s1")
cgs2.add_transition("s2", {("CarLight", "yellow"),("PedLight", "walk")},      "s4")
cgs2.add_transition("s2", {("CarLight", "yellow"),("PedLight", "dontWalk")},  "s2")
cgs2.add_transition("s2", {("CarLight", "green"), ("PedLight", "walk")},      "s4")
cgs2.add_transition("s2", {("CarLight", "green"), ("PedLight", "dontWalk")},  "s3")

cgs2.add_transition("s3", {("CarLight", "red"),   ("PedLight", "walk")},      "s0")
cgs2.add_transition("s3", {("CarLight", "red"),   ("PedLight", "dontWalk")},  "s1")
cgs2.add_transition("s3", {("CarLight", "yellow"),("PedLight", "walk")},      "s4")
cgs2.add_transition("s3", {("CarLight", "yellow"),("PedLight", "dontWalk")},  "s2")
cgs2.add_transition("s3", {("CarLight", "green"), ("PedLight", "walk")},      "s4")
cgs2.add_transition("s3", {("CarLight", "green"), ("PedLight", "dontWalk")},  "s3")

cgs2.add_transition("s4", {("CarLight", "red"),   ("PedLight", "walk")},      "s0")
cgs2.add_transition("s4", {("CarLight", "red"),   ("PedLight", "dontWalk")},  "s1")
cgs2.add_transition("s4", {("CarLight", "yellow"),("PedLight", "walk")},      "s5")
cgs2.add_transition("s4", {("CarLight", "yellow"),("PedLight", "dontWalk")},  "s2")
cgs2.add_transition("s4", {("CarLight", "green"), ("PedLight", "walk")},      "s5")
cgs2.add_transition("s4", {("CarLight", "green"), ("PedLight", "dontWalk")},  "s4")

for car in ("red", "yellow", "green"):
    for ped in ("walk", "dontWalk"):
        cgs2.add_transition("s5", {("CarLight", car), ("PedLight", ped)}, "s5")


cgs3 = CGS()

for p in (
    "idle", "airborne", "en_route", "deliver", "returning", "landed",
    "battery_low", "battery_ok", "gps_ok", "gps_lost", "no_fly_zone",
    "safe_zone", "package_onboard", "package_delivered", "package_lost",
    "emergency", "obstacle_detected", "clear_path", "mission_complete",
    "charging"
):
    cgs3.add_proposition(p)

cgs3.add_agent("Drone")
cgs3.add_agent("Package")

cgs3.add_decisions("Drone", {"fly", "hover", "land"})
cgs3.add_decisions("Package", {"attached", "released"})

for st in ("s0","s1","s2","s3","s4","s5","s6","s7","s8"):
    cgs3.add_state(st)

cgs3.set_initial_state("s0")

cgs3.label_state("s0", {"idle","landed","safe_zone","battery_ok","gps_ok",
                        "package_onboard","clear_path"})
cgs3.label_state("s1", {"airborne","en_route","battery_ok","gps_ok",
                        "package_onboard","clear_path"})
cgs3.label_state("s2", {"airborne","deliver","battery_ok","gps_ok",
                        "package_onboard","safe_zone"})
cgs3.label_state("s3", {"airborne","returning","battery_ok","gps_ok",
                        "package_delivered","clear_path"})
cgs3.label_state("s4", {"airborne","returning","battery_ok","gps_ok",
                        "package_delivered","clear_path"})
cgs3.label_state("s5", {"landed","mission_complete","charging","safe_zone",
                        "battery_ok","package_delivered"})
cgs3.label_state("s6", {"airborne","no_fly_zone","emergency","obstacle_detected",
                        "battery_ok","package_lost"})
cgs3.label_state("s7", {"airborne","gps_lost","emergency","battery_ok",
                        "package_onboard"})
cgs3.label_state("s8", {"airborne","battery_low","emergency","gps_ok",
                        "package_onboard"})

cgs3.add_transition("s0", {("Drone", "fly"),   ("Package", "attached")},  "s1")
cgs3.add_transition("s0", {("Drone", "fly"),   ("Package", "released")},  "s0")
cgs3.add_transition("s0", {("Drone", "hover"), ("Package", "attached")},  "s0")
cgs3.add_transition("s0", {("Drone", "hover"), ("Package", "released")},  "s0")
cgs3.add_transition("s0", {("Drone", "land"),  ("Package", "attached")},  "s0")
cgs3.add_transition("s0", {("Drone", "land"),  ("Package", "released")},  "s0")

cgs3.add_transition("s1", {("Drone", "fly"),   ("Package", "attached")},  "s1")
cgs3.add_transition("s1", {("Drone", "fly"),   ("Package", "released")},  "s6")
cgs3.add_transition("s1", {("Drone", "hover"), ("Package", "attached")},  "s2")
cgs3.add_transition("s1", {("Drone", "hover"), ("Package", "released")},  "s3")
cgs3.add_transition("s1", {("Drone", "land"),  ("Package", "attached")},  "s8")
cgs3.add_transition("s1", {("Drone", "land"),  ("Package", "released")},  "s3")

cgs3.add_transition("s2", {("Drone", "fly"),   ("Package", "attached")},  "s1")
cgs3.add_transition("s2", {("Drone", "fly"),   ("Package", "released")},  "s3")
cgs3.add_transition("s2", {("Drone", "hover"), ("Package", "attached")},  "s2")
cgs3.add_transition("s2", {("Drone", "hover"), ("Package", "released")},  "s3")
cgs3.add_transition("s2", {("Drone", "land"),  ("Package", "attached")},  "s6")
cgs3.add_transition("s2", {("Drone", "land"),  ("Package", "released")},  "s3")

cgs3.add_transition("s3", {("Drone", "fly"),   ("Package", "attached")},  "s6")
cgs3.add_transition("s3", {("Drone", "fly"),   ("Package", "released")},  "s4")
cgs3.add_transition("s3", {("Drone", "hover"), ("Package", "attached")},  "s6")
cgs3.add_transition("s3", {("Drone", "hover"), ("Package", "released")},  "s3")
cgs3.add_transition("s3", {("Drone", "land"),  ("Package", "attached")},  "s6")
cgs3.add_transition("s3", {("Drone", "land"),  ("Package", "released")},  "s4")

cgs3.add_transition("s4", {("Drone", "fly"),   ("Package", "attached")},  "s6")
cgs3.add_transition("s4", {("Drone", "fly"),   ("Package", "released")},  "s4")
cgs3.add_transition("s4", {("Drone", "hover"), ("Package", "attached")},  "s7")
cgs3.add_transition("s4", {("Drone", "hover"), ("Package", "released")},  "s4")
cgs3.add_transition("s4", {("Drone", "land"),  ("Package", "attached")},  "s6")
cgs3.add_transition("s4", {("Drone", "land"),  ("Package", "released")},  "s5")

cgs3.add_transition("s5", {("Drone", "fly"),   ("Package", "attached")},  "s1")
cgs3.add_transition("s5", {("Drone", "fly"),   ("Package", "released")},  "s1")
cgs3.add_transition("s5", {("Drone", "hover"), ("Package", "attached")},  "s5")
cgs3.add_transition("s5", {("Drone", "hover"), ("Package", "released")},  "s5")
cgs3.add_transition("s5", {("Drone", "land"),  ("Package", "attached")},  "s5")
cgs3.add_transition("s5", {("Drone", "land"),  ("Package", "released")},  "s5")

for d in ("fly","hover","land"):
    for p in ("attached","released"):
        cgs3.add_transition("s6", {("Drone", d), ("Package", p)}, "s6")

for d,p in (("land","released"),):
    cgs3.add_transition("s7", {("Drone", d), ("Package", p)}, "s8")
for d in ("fly","hover","land"):
    for p in ("attached","released"):
        if not (d=="land" and p=="released"):
            cgs3.add_transition("s7", {("Drone", d), ("Package", p)}, "s7")

cgs3.add_transition("s8", {("Drone", "land"), ("Package", "released")}, "s5")
cgs3.add_transition("s8", {("Drone", "land"), ("Package", "attached")}, "s6")
for d in ("fly","hover"):
    for p in ("attached","released"):
        cgs3.add_transition("s8", {("Drone", d), ("Package", p)}, "s8")


cgs4 = CGS()

for p in (
    "robot_zone1", "robot_zone2", "robot_zone3",
    "robot_idle", "robot_holding", "robot_empty",
    "conveyor_forward", "conveyor_reverse", "conveyor_stop",
    "item_on_conv", "item_at_zone1", "item_at_zone2", "item_at_zone3",
    "item_sorted", "package_lost",
    "sensor_clear", "sensor_blocked", "obstacle_detected",
    "jammed", "collision", "conveyor_overload",
    "battery_low", "battery_ok", "arm_overheat",
    "maintenance_mode", "manual_override",
    "emergency_stop", "shutdown",
    "goal_reached", "charging"
):
    cgs4.add_proposition(p)

cgs4.add_agent("RobotArm")
cgs4.add_agent("Conveyor")

cgs4.add_decisions("RobotArm",
                   {"move_left", "move_right", "stay", "pick", "drop"})
cgs4.add_decisions("Conveyor",
                   {"forward", "reverse", "stop"})

for st in ("s0","s1","s2","s3","s4","s5","s6","s7","s8","s9","s10","s11"):
    cgs4.add_state(st)

cgs4.set_initial_state("s0")

cgs4.label_state("s0", {"robot_zone1","robot_empty","robot_idle",
                        "conveyor_stop","item_at_zone1",
                        "sensor_clear","battery_ok"})
cgs4.label_state("s1", {"robot_zone1","robot_holding",
                        "conveyor_stop","sensor_clear"})
cgs4.label_state("s2", {"robot_zone2","robot_holding",
                        "conveyor_stop","sensor_clear"})
cgs4.label_state("s3", {"robot_zone2","robot_holding",
                        "conveyor_forward","item_on_conv","sensor_clear"})
cgs4.label_state("s4", {"robot_zone3","robot_holding",
                        "conveyor_stop","sensor_clear"})
cgs4.label_state("s5", {"robot_zone3","robot_empty","item_sorted",
                        "goal_reached","conveyor_stop","sensor_clear"})
cgs4.label_state("s6", {"jammed","conveyor_stop","sensor_blocked"})
cgs4.label_state("s7", {"collision","emergency_stop"})
cgs4.label_state("s8", {"battery_low","conveyor_stop","charging"})
cgs4.label_state("s9", {"maintenance_mode","conveyor_stop","manual_override"})
cgs4.label_state("s10",{"conveyor_reverse","item_on_conv","sensor_clear"})
cgs4.label_state("s11",{"shutdown","emergency_stop"})

RA = ("move_left", "move_right", "stay", "pick", "drop")
CV = ("forward", "reverse", "stop")

def add_all(state, spec, default_dest):
    """Añade las 15 combinaciones (RA × CV) usando spec para los casos especiales."""
    for ra in RA:
        for cv in CV:
            dest = spec.get((ra, cv), default_dest)
            cgs4.add_transition(
                state,
                {("RobotArm", ra), ("Conveyor", cv)},
                dest
            )

add_all("s0",
    {("pick", "stop"): "s1"},
    "s0"
)

add_all("s1",
    {("move_right", "stop"): "s2",
     ("drop",       "stop"): "s0"},
    "s1"
)

add_all("s2",
    {("stay",       "forward"): "s3",
     ("move_left",  "stop"):    "s1",
     ("drop",       "forward"): "s7",
     ("stay",       "reverse"): "s10"},
    "s2"
)

add_all("s3",
    {("move_right", "stop"):    "s4",
     ("stay",       "forward"): "s6"},
    "s3"
)

add_all("s4",
    {("drop",      "stop"): "s5",
     ("move_left", "stop"): "s3"},
    "s4"
)

add_all("s5",
    {
     ("move_left", "reverse"): "s0",
     ("stay",      "stop"):    "s8"},
    "s5"
)

add_all("s6",
    {
     ("stay",      "stop"): "s9"},
    "s6"
)

add_all("s7",
    {("stay", "stop"): "s11"},
    "s7"
)

add_all("s8",
    {("stay", "stop"): "s0"},
    "s8"
)

add_all("s9",
    {("stay", "stop"): "s0"},
    "s9"
)

add_all("s10",
    {("stay", "stop"):    "s2",
     ("stay", "forward"): "s3"},
    "s10"
)

add_all("s11", {}, "s11")