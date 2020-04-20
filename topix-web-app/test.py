import requests
import argparse
import json
from core.topix_format import convert_to_topix_format



def read_json_file(path: str):
    with open(path, "r") as f:
        return json.load(f)


def topic_model(json_content, ip, port):
    response = requests.post(
        url=f"http://{ip}:{port}/topic_model", json=json_content)
    return response.json()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ip",
        type=str,
        required=True,
        help="The IP address of the server",
    )
    parser.add_argument(
        "--port",
        type=str,
        required=True,
        help="The port of the server",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="The input file path, a json list of strings (documents)",
    )


    args = parser.parse_args()

    response = topic_model(json_content=read_json_file(
        args.input), ip=args.ip, port=args.port)

    output = convert_to_topix_format(response)
    print(json.dumps(output, indent=4, ensure_ascii=False))


main()
