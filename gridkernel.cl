__kernel void gridkernel( __global double* gr, __global double* gi, __global double* cr, __global double* ci, double dr, double di, int len) {
    int idx = get_global_id(0);

    if(idx < len) {
        gi[idx] = gi[idx] + dr * ci[idx] + di * cr[idx];
        gr[idx] = gr[idx] + dr * cr[idx] - di * ci[idx];
    }
}
