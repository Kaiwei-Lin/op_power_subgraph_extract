"""Run from the repository root: python -m onnx_graph --help."""

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle

import onnx

from .core import extract_prefix, networkx_to_onnx, onnx_to_networkx, repair_node_names
from .core import check_name_outputs, _hierarchy_options
from .omc import omc_to_networkx


def main():
    parser = argparse.ArgumentParser(description="Lossless ONNX / NetworkX conversion and prefix extraction")
    sub = parser.add_subparsers(dest="command", required=True)
    for command, help_text in [
        ("convert", "Save an ONNX/OMC model as a NetworkX pickle"),
        ("extract", "Extract a /-delimited node-name prefix to ONNX"),
        ("restore", "Restore ONNX from a trusted NetworkX pickle"),
        ("repair-names", "Recover node hierarchy from tensor paths"),
    ]:
        cmd = sub.add_parser(command, help=help_text)
        cmd.add_argument("source", type=Path)
        cmd.add_argument("destination", type=Path)
        if command == "extract":
            cmd.add_argument("--prefix", required=True)
        if command in ("convert", "extract"):
            cmd.add_argument("--repair-names", action="store_true")
            cmd.add_argument("--hierarchy-source", choices=("name", "outputs"), default="name",
                             help="outputs: use output paths as hierarchy and preserve original node names")
        if command != "restore":
            cmd.add_argument("--name-aliases", type=Path, help="JSON prefix aliases: {old: new}")
        if command == "convert":
            cmd.add_argument("--ge-proto", type=Path, help="DDK ge_ir_pb2.py for OMC input")
        if command == "repair-names":
            cmd.add_argument("--report", type=Path, help="JSON report (default: destination.rename.json)")
    args = parser.parse_args()
    try:
        _, aliases = _hierarchy_options(getattr(args, "hierarchy_source", "name"), False,
                                       getattr(args, "name_aliases", None))
    except ValueError as exc:
        parser.error(str(exc))
    if args.source.resolve() == args.destination.resolve():
        parser.error("Source and destination must differ")
    is_omc = args.source.suffix.lower() in {".omc", ".om"}
    if is_omc and args.command in {"extract", "repair-names"}:
        parser.error("OMC supports convert/statistics/NetworkX prefix selection; executable ONNX export requires the original ONNX")
    report_path = None
    if args.command == "repair-names":
        report_path = args.report or args.destination.with_suffix(".rename.json")
        if report_path.resolve() in (args.source.resolve(), args.destination.resolve()):
            parser.error("Report must differ from source and destination")
    if args.command == "restore":
        # Pickle is a local Python artifact format; load trusted files only.
        with args.source.open("rb") as stream:
            graph = pickle.load(stream)
        model = networkx_to_onnx(graph)
        issues = graph.graph.get("name_output_mismatches", [])
    elif args.command == "repair-names":
        model, report = repair_node_names(onnx.load(str(args.source)), name_aliases=aliases)
        issues = check_name_outputs(model, name_aliases=aliases, rename_report=report)
        onnx.checker.check_model(model)
    else:
        graph = (omc_to_networkx(args.source, repair_names=args.repair_names, ge_proto_path=args.ge_proto,
                                 name_aliases=aliases, hierarchy_source=args.hierarchy_source)
                 if is_omc else onnx_to_networkx(args.source, repair_names=args.repair_names,
                                                name_aliases=aliases, hierarchy_source=args.hierarchy_source))
        issues = graph.graph["name_output_mismatches"]
        if args.command == "extract":
            graph = extract_prefix(graph, args.prefix)
            model = networkx_to_onnx(graph)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "convert":
        with args.destination.open("wb") as stream:
            pickle.dump(graph, stream, protocol=pickle.HIGHEST_PROTOCOL)
        summary = dict(nodes=len(graph), edges=graph.number_of_edges(),
                       rename_status=dict(Counter(r["status"] for r in graph.graph["rename_report"])))
    else:
        onnx.save(model, str(args.destination))
        summary = dict(nodes=len(model.graph.node), initializers=len(model.graph.initializer),
                       inputs=[v.name for v in model.graph.input], outputs=[v.name for v in model.graph.output])
        if report_path:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            summary["rename_status"] = dict(Counter(r["status"] for r in report))
    if args.command != "repair-names":
        summary.update(hierarchy_source=graph.graph.get("hierarchy_source", "name"),
                       output_hierarchy_issues=graph.graph.get("output_hierarchy_issues", []))
    print(json.dumps(dict(destination=str(args.destination), name_output_mismatches=issues, **summary),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
