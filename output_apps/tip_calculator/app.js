function createApp() {
  const state = { items: [], count: 0, bill: 0, rate: 0, result: 0,
                  total: 0, input: '', filter: 'all' };
  function dispatch(action, payload) {
    switch (action) {
      case "SET_BILL": { state.bill = Number(payload); break; }
      case "SET_RATE": { state.rate = Number(payload); break; }
      case "COMPUTE": { state.result = state.bill + state.bill * state.rate; break; }
      default: break;
    }
  }
  return { dispatch: dispatch, getState: function(){ return state; } };
}
if (typeof module !== 'undefined') { module.exports = { createApp: createApp }; }
