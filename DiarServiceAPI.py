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


class RespReq:
    thread_pref = "Request_"
    code_msg: Dict[int, str] = {202: "The request was accepted for processing, but it was not completed.",
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


class Utils:
    @staticmethod
    def thread_name(id_req):
        return RespReq.thread_pref + str(id_req)

    @staticmethod
    def get_args(class_name):
        parser = reqparse.RequestParser()
        if class_name == ApiBase.endpoint:
            parser.add_argument(RespReq.Field.data, type=FileStorage, location='files')
        elif class_name == ApiResult.endpoint:
            parser.add_argument(RespReq.Field.id)
            parser.add_argument(RespReq.Field.num, type=inputs.boolean)
        else:
            raise Exception('Failed to collect RequestParser()')
        return parser.parse_args()

    @staticmethod
    def build(id_req, code, msg='', field_name=None, field_pos=None):
        message = msg if msg != '' else RespReq.code_msg[code]
        if code == 400:
            message = message.format(field_name, field_pos)
        return {'id': id_req, 'message': message}, code

    @staticmethod
    def dir_received_files():
        dir_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ReceivedFiles')
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        return dir_name

    @staticmethod
    def get_ID_request(data: FileStorage):
        if data.stream.name is not None and data.filename is not None:
            id_req = os.path.split(data.stream.name)[1]
        else:
            id_req = mktemp(dir='')
        return id_req

    @staticmethod
    def get_filename(ID, fn):
        filename = ID + os.path.splitext(fn)[1].lower()
        filename = re.sub(Settings.unsupported_chars_in_filename, '', filename)
        return filename


class ProcessingRequest(threading.Thread):
    def __init__(self, id_request, audio_filedir, audio_filename):
        threading.Thread.__init__(self, name=Utils.thread_name(id_request))
        self.id = id_request
        self.status = 0
        self.filedir = audio_filedir
        self.filename = os.path.join(audio_filedir, audio_filename)
        self.res_file = ''
        self.res_num_of_speakers = 0
        self.list_errs = list()
        self.error = False

    def run(self):
        self.status = 202
        try:
            self.res_file, self.res_num_of_speakers = process(self.filename, debug_mode=DEBUG_MODE)
            self.res_file = self.res_file.replace(self.filedir + os.sep, '')
            self.status = 200
        except Exception as e:
            if DEBUG_MODE:
                print('{}: '.format(self.id), e)
            self.list_errs.append(str(e))
            self.error = True
            self.status = 500

        os.remove(self.filename)


DEBUG_MODE = True
DIC_REQS: Dict[str, ProcessingRequest] = dict()


class ApiBase(Resource):
    def post(self):
        data: FileStorage = Utils.get_args(self.endpoint)[RespReq.Field.data]

        if data is not None:
            ID = Utils.get_ID_request(data)
            filename = Utils.get_filename(ID, data.filename)
            filedir = Utils.dir_received_files()
            file_path = os.path.join(filedir, filename)
            data.save(file_path)

            res_check, msg = check_file(file_path)
            code = 200
            if res_check:
                new_prc = ProcessingRequest(ID, filedir, filename)
                DIC_REQS[ID] = new_prc
                new_prc.start()
            else:
                os.remove(file_path)
                ID = None
                code = 415
        else:
            ID = None
            code = 400
        return Utils.build(ID, code, field_name=RespReq.Field.data, field_pos=RespReq.Field.Position.body)


class ApiResult(Resource):
    def get(self):
        args = Utils.get_args(self.endpoint)
        ID = args[RespReq.Field.id]
        num_speakers = args[RespReq.Field.num]

        if ID is None:
            code = 400
            return Utils.build(ID, code, field_name=RespReq.Field.id, field_pos=RespReq.Field.Position.params)

        req = DIC_REQS.get(ID)
        if req is None:
            code = 404
        else:
            code = req.status
            if code == 200:
                req.join()
                if num_speakers is not None and num_speakers is True:
                    return Utils.build(ID, code, str(req.res_num_of_speakers))
                else:
                    return send_from_directory(req.filedir, req.res_file, as_attachment=True)

        return Utils.build(ID, code)


app = Flask(__name__)
api = Api(app)

api.add_resource(ApiBase, Settings.base_path)
api.add_resource(ApiResult, Settings.result_path)

if __name__ == '__main__':
    app.run(host=Settings.host, port=Settings.port, debug=DEBUG_MODE, use_reloader=False)
