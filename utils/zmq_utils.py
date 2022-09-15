#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import zmq


def zmq_listen(context, port, conflate=False, hwm=0):
    socket = context.socket(zmq.SUB)
    if hwm > 0:
        socket.set_hwm(hwm)
    if conflate:
        socket.setsockopt(zmq.CONFLATE, 1)
    socket.bind(f"tcp://127.0.0.1:{port}")
    socket.subscribe('')
    return socket


def zmq_connect(context, port, conflate=False, hwm=0):
    socket = context.socket(zmq.PUB)
    if hwm > 0:
        socket.set_hwm(hwm)
    if conflate:
        socket.setsockopt(zmq.CONFLATE, 1)
    socket.connect(f'tcp://127.0.0.1:{port}')
    return socket