import os
import sys

sys.path.append('/srv/AcornProject/Investar_final')
os.environ.setdefault('PYTHON_EGG_CACHE', '/srv/AcornProject/Investar_final/egg_cache')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Investar_final.settings')
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
