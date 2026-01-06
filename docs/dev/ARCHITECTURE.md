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
