"""
=============================
MPE timing (:mod:`mango.mpe`)
=============================

.. currentmodule:: mango.mpe

Functions
==========

.. autosummary::
   :toctree: generated/

   init_log - Initialise MPE logging.
   start_log - Start logging.
   stop_log - Stop logging.
   finish_log - Calculates/gathers timing stats an logs message.
   create_state - Creates a logging state.
   log_event - Notifies a timing event.
   get_time_since_start - Returns accumulated event time since creation. 
"""

import mango.core
from mango.core._mango_open_core import _MPE_Init_log              as init_log
from mango.core._mango_open_core import _MPE_Start_log             as start_log
from mango.core._mango_open_core import _MPE_Stop_log              as stop_log
from mango.core._mango_open_core import _MPE_Finish_log            as finish_log
from mango.core._mango_open_core import _MPE_Create_state          as create_state
from mango.core._mango_open_core import _MPE_Log_event             as log_event
from mango.core._mango_open_core import _MPE_get_time_since_start  as get_time_since_start

__all__ = [s for s in dir() if not s.startswith('_')]


