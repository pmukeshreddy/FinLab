from dataclasses import dataclass


@dataclass
class TaskConfig:
    org_name: str


@dataclass
class TaskExportConfig:
    export_path: str
    templates_path: str | None
    task_config: TaskConfig
