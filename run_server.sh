#!/usr/bin/bash

gnome-terminal --command "python manage.py"
sleep 2
gnome-terminal --command "python manage.py --mode client --client_node client1 --client_loader 0"
sleep 3
gnome-terminal --command "python manage.py --mode client --client_node client2 --client_loader 1"
sleep 5
gnome-terminal --command "python manage.py --mode client --client_node client3 --client_loader 2"