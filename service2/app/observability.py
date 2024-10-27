""" initialize observability tools and configurations """

from fastapi import FastAPI
from fastapi.routing import APIRoute
from prometheus_client import Gauge, REGISTRY
from starlette_exporter import PrometheusMiddleware, handle_metrics
from starlette.responses import Response  # Ensure this is imported
from app.config import logger


def init_observability(app: FastAPI) -> None:
    """
    Initialize observability tools and configurations.
    This function should be called at the beginning of the application
    to set up observability tools and configurations.
    """
    # Initialize Prometheus and other observability tools
    init_prometheus(app)

    # Initialize other observability tools if necessary
    # e.g., init_zipkin_tracer(), init_grafana_dashboard()


def init_prometheus(app: FastAPI) -> None:
    """
    Initialize Prometheus observability.
    This includes setting up middleware and the `/metrics` route.
    """
    # Define the Prometheus metrics route explicitly for OpenAPI documentation
    metrics_route = APIRoute(
        path="/metrics",
        endpoint=handle_metrics,  # No need for lambda, use handle_metrics directly
        methods=["GET"],
        summary="Prometheus Metrics",
        description="Exposes Prometheus metrics for monitoring.",
        response_class=Response,
    )

    # Add the metrics route to FastAPI
    app.router.routes.append(metrics_route)

    # Add Prometheus middleware to collect metrics for each request
    app.add_middleware(PrometheusMiddleware)

    # Custom application info as a Gauge metric (static info)
    try:
        logger.debug("Registering custom metrics: application_info")
        info = Gauge(
            "application_info",
            "Agentic GrapRAG Chatbot application info",
            ["version", "environment"],
            registry=REGISTRY,
        )
        # Set the static values for your application
        info.labels(version="1.0.0", environment="production").set(1)
    except ValueError as e:
        if "Duplicated timeseries" in str(e):
            logger.warning("Metric 'application_info' is already registered: %s", e)
