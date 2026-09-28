"""MCP stdio server for Robot EKF."""
import sys
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from client import RobotEKF

ekf = RobotEKF()

def handle_rpc(request):
    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params", {})

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "ekf_predict",
                        "description": "Execute motion prediction step with velocity v, angular velocity omega, dt",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "v": {"type": "number"},
                                "omega": {"type": "number"},
                                "dt": {"type": "number"}
                            },
                            "required": ["v", "omega", "dt"]
                        }
                    },
                    {
                        "name": "ekf_update",
                        "description": "Execute measurement update step with landmark position and range/bearing observation",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "landmark": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                                "measurement": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
                            },
                            "required": ["landmark", "measurement"]
                        }
                    },
                    {
                        "name": "ekf_get_state",
                        "description": "Retrieve current estimated state and covariance trace",
                        "inputSchema": {"type": "object"}
                    }
                ]
            }
        }
    elif method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {})
        if name == "ekf_predict":
            ekf.predict(float(args.get("v", 0.0)), float(args.get("omega", 0.0)), float(args.get("dt", 0.1)))
            return {"jsonrpc": "2.0", "id": req_id, "result": ekf.get_state()}
        elif name == "ekf_update":
            lm = tuple(args.get("landmark"))
            meas = tuple(args.get("measurement"))
            ekf.update(lm, meas)
            return {"jsonrpc": "2.0", "id": req_id, "result": ekf.get_state()}
        elif name == "ekf_get_state":
            return {"jsonrpc": "2.0", "id": req_id, "result": ekf.get_state()}
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method {name} not found"}}
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32600, "message": "Invalid request"}}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_rpc(req)
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "error": {"code": -32700, "message": str(e)}}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
