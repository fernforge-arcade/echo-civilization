function createApp() {
  const state = { items: [], count: 0, bill: 0, rate: 0, result: 0,
                  total: 0, input: '', filter: 'all' };
  function dispatch(action, payload) {
    switch (action) {
      case "INC": { state.count += 1; break; }
      case "DEC": { state.count -= 1; break; }
      case "RESET": { state.count = 0; break; }
      default: break;
    }
  }
  return { dispatch: dispatch, getState: function(){ return state; } };
}
if (typeof module !== 'undefined') { module.exports = { createApp: createApp }; }
