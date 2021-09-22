#!/usr/bin/bash

gnome-terminal --command "python manage.py"
sleep 2
gnome-terminal --command "python manage.py --mode client --client_node client1 --client_loader 0"
gnome-terminal --command "python manage.py --mode client --client_node client2 --client_loader 1"