import sys
path = '/home/mikekestemont/echoes'
if path not in sys.path:
   sys.path.insert(0, path)

from api import app as application
if __name__ == '__main__':
    application.run(debug=True, host='127.0.0.1', port=5000)