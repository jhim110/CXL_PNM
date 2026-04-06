# Native:  make
# AArch64 (conda-forge / mamba: gcc_linux-aarch64 등):
#          make aarch64
#          make CROSS_COMPILE=aarch64-conda-linux-gnu-
# AArch64 (apt, sudo 필요): make aarch64-apt
#          sudo apt install gcc-aarch64-linux-gnu

CROSS_COMPILE ?=
# 명령행에서 CC=... 를 주면 그걸 우선
CC              := $(CROSS_COMPILE)gcc
CFLAGS          ?= -Wall -Wextra -O3
LDFLAGS         ?= 
LDLIBS          ?= -lm

TARGET_NATIVE  := test_attention
TARGET_AARCH64 := test_attention_aarch64
TARGET_AARCH64_SVE := test_attention_aarch64_sve
TARGET_AARCH64_SVE2 := test_attention_aarch64_sve2
TARGET_AARCH64_SME := test_attention_aarch64_sme
TARGET_AARCH64_SME2 := test_attention_aarch64_sme2
TARGET         ?= $(TARGET_NATIVE)
SRC            := attention_test.c

# AArch64에서 SVE/SVE2를 켠 채로 빌드하려면 여기에 -march를 추가.
AARCH64_CFLAGS ?=

.PHONY: all clean run aarch64 aarch64-apt aarch64-sve aarch64-sve2 aarch64-sme aarch64-sme2

all: $(TARGET)

# 핵심 수정 부분: $(LDLIBS)를 명령어의 가장 마지막에 배치
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) $< -o $@ $(LDLIBS)

# conda-forge 크로스 컴파일러
aarch64:
	$(MAKE) CROSS_COMPILE=aarch64-conda-linux-gnu- TARGET=$(TARGET_AARCH64) CFLAGS='$(CFLAGS) $(AARCH64_CFLAGS)'

aarch64-sve:
	$(MAKE) CROSS_COMPILE=aarch64-conda-linux-gnu- TARGET=$(TARGET_AARCH64_SVE) CFLAGS='$(CFLAGS) -march=armv9-a+sve'

aarch64-sve2:
	$(MAKE) CROSS_COMPILE=aarch64-conda-linux-gnu- TARGET=$(TARGET_AARCH64_SVE2) CFLAGS='$(CFLAGS) -march=armv9-a+sve+sve2'

aarch64-sme:
	$(MAKE) CROSS_COMPILE=aarch64-conda-linux-gnu- TARGET=$(TARGET_AARCH64_SME) CFLAGS='$(CFLAGS) -march=armv9-a+sme'

aarch64-sme2:
	$(MAKE) CROSS_COMPILE=aarch64-conda-linux-gnu- TARGET=$(TARGET_AARCH64_SME2) CFLAGS='$(CFLAGS) -march=armv9-a+sve2+sme2'

aarch64-apt:
	$(MAKE) CROSS_COMPILE=aarch64-linux-gnu- TARGET=$(TARGET_AARCH64) CFLAGS='$(CFLAGS) $(AARCH64_CFLAGS)'

clean:
	rm -f $(TARGET_NATIVE) $(TARGET_AARCH64) $(TARGET_AARCH64_SVE) $(TARGET_AARCH64_SVE2) $(TARGET_AARCH64_SME) $(TARGET_AARCH64_SME2)

run: $(TARGET)
	./$(TARGET) $(ARGS)