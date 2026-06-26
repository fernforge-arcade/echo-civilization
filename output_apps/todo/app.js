function createApp() {
  const state = { items: [], count: 0, bill: 0, rate: 0, result: 0,
                  total: 0, input: '', filter: 'all' };
  function dispatch(action, payload) {
    switch (action) {
      case "ADD": { state.items.push({text: payload, done: false}); break; }
      case "REMOVE": { state.items.splice(payload, 1); break; }
      case "TOGGLE": { if (state.items[payload]) state.items[payload].done = !state.items[payload].done; break; }
      default: break;
    }
  }
  return { dispatch: dispatch, getState: function(){ return state; } };
}
if (typeof module !== 'undefined') { module.exports = { createApp: createApp }; }
