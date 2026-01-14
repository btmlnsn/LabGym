# LabGym - Architecture Guard-rails (Draft)

The codebase is divided into concentric layers


Imports may only go **inward**, never outward.

    Layer      |    May Import
--------------------------------
UI             | app, domain, config
App            | subsystems, domain, config
Subsystems     | subsystems.shared, domain, config
Domain, Config | stdlib (NO wx, tensorflow, torch, cv2, detectron2, ...)

Violations fail CI via 'import-linter' (will add at the end).
