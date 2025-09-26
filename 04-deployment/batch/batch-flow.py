from datetime import datetime, timezone
import subprocess
from prefect import flow, task


@task(log_prints=True)
def run_docker(year: int, month: int, taxi_type: str):
    cmd = [
        "docker", "run", "--rm",
        "batch-pred",
        "--year", str(year),
        "--month", f"{month:02d}",
        "--taxi_type", taxi_type,
    ]
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


@flow(name="monthly-batch-pred", log_prints=True)
def run_batch_pred(taxi_type: str = "yellow", year: int | None = None, month: int | None = None):
    now = datetime.now(timezone.utc)
    if year is None:
        year = now.year
    if month is None:
        month = now.month

    run_docker(year, month, taxi_type)


if __name__ == "__main__":
    # 🚨 make sure you have created a Docker work pool first:
    # prefect work-pool create --type docker my-docker-pool
    #
    # And have a worker running:
    # prefect worker start --pool my-docker-pool --name docker-worker

    run_batch_pred.deploy(
        name="monthly-batch-pred",
        work_pool_name="my-docker-pool",   # <-- your docker work pool name
        image="batch-pred",         # <-- your local Docker image
        push=False,                        # don't try to push image, use local one
        cron="0 0 1 * *",                  # schedule: 1st of every month at 00:00 UTC
        parameters={"taxi_type": "yellow"},
    )
