################################################################################
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

CC=g++
CFLAGS:= -Wall -std=c++11
CFLAGS+= -shared -fPIC -fpermissive

APP:= deepstream-app

TARGET_DEVICE = $(shell $(CC) -dumpmachine | cut -f1 -d -)

NVDS_VERSION:=5.0
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

ifeq ($(TARGET_DEVICE),aarch64)
  CFLAGS+= -DPLATFORM_TEGRA
endif

SRCS:= $(wildcard ../src/*.c)
SRCS+= $(wildcard /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/apps/apps-common/src/*.c)

INCS:= $(wildcard ../src/*.h)

PKGS:= gstreamer-1.0 gstreamer-video-1.0 x11 json-glib-1.0

OBJS:= $(SRCS:.c=.o)

CFLAGS+= -I./ -I/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/apps/apps-common/includes -I/opt/nvidia/deepstream/deepstream/sources/includes -DDS_VERSION_MINOR=0 -DDS_VERSION_MAJOR=5 \
	 -I/usr/local/cuda/include

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvdsgst_smartrecord -lnvds_utils -lnvds_msgbroker -lm \
       -lgstrtspserver-1.0 -ldl -Wl,-rpath,$(LIB_INSTALL_DIR) \
       -L/usr/local/cuda/lib64/ -lcudart -lcublasLt \

CFLAGS+= `pkg-config --cflags $(PKGS)`

LIBS+= `pkg-config --libs $(PKGS)`

all: $(APP)

%.o: %.c $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CC) -o $(APP) $(OBJS) $(LIBS)

install: $(APP)

clean:
	rm -rf $(OBJS) $(APP)
