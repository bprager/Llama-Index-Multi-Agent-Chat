""" initialize observability tools and configurations """

from prometheus_flask_exporter import PrometheusMetrics  # type: ignore [import-untyped]
from fastapi import FastAPI


def init_observability(app: FastAPI) -> PrometheusMetrics:
    """
    Initialize observability tools and configurations.
    This function should be called at the beginning of the application
    to set up observability tools and configurations.
    """
    # Initialize Zipkin tracer
    # init_zipkin_tracer()

    # Initialize Prometheus metrics
    metrics: PrometheusMetrics = PrometheusMetrics(app)

    # Initialize Grafana dashboard
    # init_grafana_dashboard()

    # Initialize other observability tools as needed
    return metrics
