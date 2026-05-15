import json

from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .services import (
    FEATURE_ORDER,
    ProjectDependencyError,
    get_sample_choices,
    get_sample_features,
    get_plot_path,
    get_recent_transactions,
    load_dashboard_metrics,
    predict_transaction,
)


def dashboard(request):
    context = {"active_page": "dashboard"}
    try:
        context["metrics"] = load_dashboard_metrics(request.GET)
        context["recent_predictions"] = get_recent_transactions(limit=5)
    except (FileNotFoundError, ProjectDependencyError) as exc:
        context["setup_error"] = str(exc)
    return render(request, "fraud_app/dashboard.html", context)


@require_http_methods(["GET", "POST"])
def predict(request):
    result = None
    form_values = _default_form_values()
    error = None
    selected_sample = request.GET.get("sample") or request.POST.get("selected_sample", "")
    loaded_sample = None

    try:
        sample_choices = get_sample_choices()
        loaded_sample = get_sample_features(selected_sample)
        if loaded_sample and request.method == "GET":
            form_values.update(loaded_sample["features"])
    except (FileNotFoundError, ProjectDependencyError) as exc:
        sample_choices = []
        error = str(exc)

    if request.method == "POST":
        form_values.update({name: request.POST.get(name, 0) for name in FEATURE_ORDER})
        try:
            result = predict_transaction(request.POST)
            form_values.update(result["features"])
        except (ValueError, ProjectDependencyError, FileNotFoundError) as exc:
            error = str(exc)

    return render(
        request,
        "fraud_app/predict.html",
        {
            "active_page": "predict",
            "feature_order": FEATURE_ORDER,
            "sample_choices": sample_choices,
            "selected_sample": selected_sample,
            "loaded_sample": loaded_sample,
            "v_feature_fields": [
                {"name": name, "value": form_values.get(name, 0)}
                for name in [f"V{i}" for i in range(1, 29)]
            ],
            "form_values": form_values,
            "result": result,
            "error": error,
        },
    )


def history(request):
    return render(
        request,
        "fraud_app/history.html",
        {
            "active_page": "history",
            "transactions": get_recent_transactions(limit=50),
        },
    )


@csrf_exempt
@require_http_methods(["POST"])
def predict_api(request):
    try:
        payload = json.loads(request.body.decode("utf-8"))
        result = predict_transaction(payload)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)
    except (ValueError, ProjectDependencyError, FileNotFoundError) as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    return JsonResponse(result)


def plot_image(request, name):
    path = get_plot_path(name)
    if not path:
        raise Http404("Plot not found.")
    return FileResponse(open(path, "rb"), content_type="image/png")


def _default_form_values():
    defaults = {name: 0 for name in FEATURE_ORDER}
    defaults["Amount"] = 120.0
    return defaults
