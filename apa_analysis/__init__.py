# Backward-compat alias so pickles saved under the old module path
# (Analysis.Characterisation_v2.*) can be loaded after the refactor.
import sys
import types
import apa_analysis.Characterisation as _char
import apa_analysis.Characterisation.DataClasses as _dc

# Register the old path as an alias for the new one
sys.modules['Analysis'] = types.ModuleType('Analysis')
sys.modules['Analysis.Characterisation_v2'] = _char
sys.modules['Analysis.Characterisation_v2.DataClasses'] = _dc
