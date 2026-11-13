
from datetime import datetime, timezone

import psutil
import speedtest


def get_ram_usage() -> dict:
    """
    Retrieve physical RAM usage statistics
    for the machine running this process.
    """

    memory = psutil.virtual_memory()

    return {
        "total_gb": round(memory.total / (1024 ** 3), 2),
        "used_gb": round(memory.used / (1024 ** 3), 2),
        "available_gb": round(memory.available / (1024 ** 3), 2),
        "usage_percent": memory.percent,
    }


def get_cpu_usage() -> dict:
    """
    Retrieve CPU utilization and processor information
    for the machine running this process.
    """

    cpu_percent = psutil.cpu_percent(interval=1)

    return {
        "usage_percent": cpu_percent,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
    }


def get_internet_speed() -> dict:
    """
    Measure internet download speed, upload speed,
    and ping using Ookla Speedtest infrastructure.
    """

    tester = speedtest.Speedtest()
    tester.get_best_server()

    download_speed = tester.download()
    upload_speed = tester.upload()

    ping = tester.results.ping

    return {
        "download_mbps": round(download_speed / 1_000_000, 2),
        "upload_mbps": round(upload_speed / 1_000_000, 2),
        "ping_ms": round(ping, 2),
    }


def get_system_uptime() -> dict:
    """
    Retrieve the uptime of the machine running
    this process, based on its last boot time.
    """

    boot_timestamp = psutil.boot_time()

    current_timestamp = datetime.now(timezone.utc).timestamp()

    uptime_seconds = int(current_timestamp - boot_timestamp)

    days, remainder = divmod(uptime_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    boot_time = datetime.fromtimestamp(
        boot_timestamp,
        tz=timezone.utc
    ).astimezone()

    return {
        "uptime_seconds": uptime_seconds,
        "days": days,
        "hours": hours,
        "minutes": minutes,
        "seconds": seconds,
        "last_boot_time": boot_time.isoformat(timespec="seconds"),
    }
