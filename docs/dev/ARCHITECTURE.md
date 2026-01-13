# LabGym Architecture (Draft)

Layers
-------

GUI (wxPython)
└─ app.registry           – GUI ⇄ workflow routing
   └─ workflows.<step>    – preprocessing, training, …
      └─ domain           – pure domain objects
         └─ io            – FS / video / spreadsheets
            └─ utils      – generic helpers
      └─ ml               – low-level ML (Detectron2, …)


- GUI - where wxPython lives and why it is a thin shell
- Workflows - orchestrates end-to-end tasks, no ML logic inside
- Domain - pure business objects, 100% test-covered
- ML - third-party libraries wrapped (Detectron2, Keras, ...)
- IO - file/video helpers (FS-only, no network)
- Utils - truly generic helpers, no LabGym imports


Dependency Rules:
layer            may import                must NOT import
--------------   -----------------------   -----------------------------
GUI              workflows, domain, utils  ml
workflows        domain, ml, utils         gui
domain           utils                     gui, ml
ml               utils                     gui, workflows, domain
io               utils                     gui
utils            (none)                    (none)