import glob
import json
import os
import re
import threading
from tempfile import mktemp
from typing import Dict

from flask import Flask, send_from_directory
from flask_restful import Resource, Api, reqparse, inputs
from werkzeug.datastructures import FileStorage

from DiarService import process, check_file


class Settings:
    host = '127.0.0.1'
    port = 5000
    base_path = '/'
    result_path = '/result'
    unsupported_chars_in_filename = r'[/:*?"<>\\|]'


class Response:
    code_msg: Dict[int, str] = {202: "The request was accepted for processing, but it was not completed.",
                                201: "Request completed.",
                                200: "OK",
                                400: "Error in the request: there is no '{0}' field in {1}.",
                                404: "The specified request ID was not found.",
                                500: "An unexpected error occurred while processing the file. "
                                     "You can try again or make another request."}

    class Field:
        id = 'id'
        data = 'data'
        num = 'num_speakers'

        class Position:
            body = 'body'
            params = 'parameters'

    @staticmethod
    def build(id_req, code, msg=None, field_name=None, field_pos=None):
        message = msg if msg is not None else Response.code_msg[code]
        if code == 400:
            message = message.format(field_name, field_pos)
        return {'id': id_req, 'message': message}, code


class Request:
    thread_pref = "Request_"

    def __init__(self, status, num=0, msg=None):
        self.status = status
        self.num_speakers = num
        self.message = msg

    def to_json(self, ID):
        json_file = os.path.join(Utils.dir_received_files(), ID + Utils.format_info)
        with open(json_file, 'w') as f:
            json.dump(self.__dict__, f)

    @staticmethod
    def from_json(json_file):
        with open(json_file) as f:
            json_dict = json.load(f)
        return Request(json_dict['status'], json_dict['num_speakers'], json_dict['message'])

    @staticmethod
    def thread_name(id_req):
        return Request.thread_pref + str(id_req)

    @staticmethod
    def get_args(class_name):
        parser = reqparse.RequestParser()
        if class_name == ApiBase.endpoint:
            parser.add_argument(Response.Field.data, type=FileStorage, location='files')
        elif class_name == ApiResult.endpoint:
            parser.add_argument(Response.Field.id)
            parser.add_argument(Response.Field.num, type=inputs.boolean)
        else:
            raise Exception('Failed to collect RequestParser()')
        return parser.parse_args()

    @staticmethod
    def get_ID_request(data: FileStorage):
        if data.stream.name is not None and data.filename is not None:
            id_req = os.path.split(data.stream.name)[1]
        else:
            id_req = mktemp(dir='')
        return id_req

    @staticmethod
    def get_request_info(ID):
        info_file = os.path.join(Utils.dir_received_files(), ID + Utils.format_info)
        req = None
        if os.path.isfile(info_file):
            req = Request.from_json(info_file)
        return req

    @staticmethod
    def check_previous():
        dir_files = Utils.dir_received_files()
        os.chdir(dir_files)
        types = ('*{}'.format(Utils.format_info), '*{}'.format(Utils.format_result))
        res_inf_files = []
        for ext in types:
            res_inf_files.extend(glob.glob(ext))
        list_files = os.listdir(dir_files)
        files = [x for x in list_files if x not in res_inf_files]
        for file in files:
            ID = os.path.splitext(file)[0]
            filename = os.path.join(dir_files, file)
            res_check, msg = check_file(filename)
            req = Request(200)
            if res_check:
                req.to_json(ID)
                new_prc = ProcessingRequest(ID, filename)
                new_prc.start()
            else:
                req.status = 415
                req.message = msg
                req.to_json(ID)


class Utils:
    format_audio = '.wav'
    format_info = '.json'
    format_result = '.csv'

    @staticmethod
    def dir_files(dn):
        dir_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), dn)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        return dir_name

    @staticmethod
    def dir_received_files():
        return Utils.dir_files('ReceivedFiles')

    @staticmethod
    def get_filename(ID, fn):
        filename = ID + os.path.splitext(fn)[1].lower()
        filename = re.sub(Settings.unsupported_chars_in_filename, '', filename)
        return filename


class ProcessingRequest(threading.Thread):
    def __init__(self, id_request, audio_filename):
        threading.Thread.__init__(self, name=Request.thread_name(id_request))
        self.ID = id_request
        self.filename = audio_filename
        self.error = False
        self.error_str = None
        self.request = Request(200)

    def run(self):
        self.request.status = 202
        self.request.to_json(self.ID)
        try:
            _, num_of_speakers = process(self.filename, debug_mode=DEBUG_MODE)
            self.request.num_speakers = int(num_of_speakers)
        except Exception as e:
            if DEBUG_MODE:
                print('{}: '.format(self.ID), e)
            self.error_str = str(e)
            self.error = True
            self.request.status = 500
        else:
            self.request.status = 201
        finally:
            os.remove(self.filename)
            self.request.to_json(self.ID)


class ApiBase(Resource):
    def post(self):
        data: FileStorage = Request.get_args(self.endpoint)[Response.Field.data]
        msg = None
        ID = None
        code = 200
        if data is not None:
            ID = Request.get_ID_request(data)
            filename = Utils.get_filename(ID, data.filename)
            filedir = Utils.dir_received_files()
            file_path = os.path.join(filedir, filename)
            data.save(file_path)

            res_check, msg = check_file(file_path)
            if res_check:
                Request(code).to_json(ID)
                new_prc = ProcessingRequest(ID, file_path)
                new_prc.start()
            else:
                os.remove(file_path)
                ID = None
                code = 415
        else:
            code = 400
        return Response.build(ID, code, msg=msg, field_name=Response.Field.data, field_pos=Response.Field.Position.body)


class ApiResult(Resource):
    def get(self):
        args = Request.get_args(self.endpoint)
        ID = args[Response.Field.id]
        msg = None

        if ID is not None:
            num_speakers = args[Response.Field.num]
            req = Request.get_request_info(ID)
            if req is not None:
                code = req.status
                msg = req.message
                if code == 201:
                    if num_speakers is not None and num_speakers is True:
                        return Response.build(ID, code, str(req.num_speakers))
                    else:
                        return send_from_directory(Utils.dir_received_files(),
                                                   ID + Utils.format_result,
                                                   as_attachment=True)
            else:
                code = 404
        else:
            code = 400

        return Response.build(ID, code, msg, field_name=Response.Field.id, field_pos=Response.Field.Position.params)


app = Flask(__name__)
api = Api(app)

api.add_resource(ApiBase, Settings.base_path)
api.add_resource(ApiResult, Settings.result_path)

DEBUG_MODE = True


def main():
    Request.check_previous()
    app.run(host=Settings.host, port=Settings.port, debug=DEBUG_MODE, use_reloader=False)


if __name__ == '__main__':
    main()
