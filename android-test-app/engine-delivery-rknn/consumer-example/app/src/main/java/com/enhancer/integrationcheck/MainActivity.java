package com.enhancer.integrationcheck;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import com.enhancer.rknn.RknnEnhancer;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Standalone AAR consumer. No test-app or SDK project dependency. */
public final class MainActivity extends Activity {
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private RknnEnhancer enhancer;
    private TextView status;
    private ImageView image;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        status = new TextView(this);
        status.setText("Preparing RKNN model...");
        image = new ImageView(this);
        image.setAdjustViewBounds(true);
        layout.addView(status);
        layout.addView(image);
        setContentView(layout);
        worker.execute(this::runSample);
    }

    private void runSample() {
        try {
            File model = new File(getFilesDir(), "models/GFPGANv1.4_rgb_bias_consolidated_rk3576.rknn");
            if (!model.isFile()) {
                throw new IllegalStateException("Provision the .rknn model at " + model.getAbsolutePath());
            }
            // These landmarks belong only to the bundled sample. A real host app
            // must detect five landmarks for its own oriented/resized bitmap.
            JSONObject metadata;
            try (InputStream in = getAssets().open("sample.json")) {
                ByteArrayOutputStream bytes = new ByteArrayOutputStream();
                byte[] buffer = new byte[4096];
                int count;
                while ((count = in.read(buffer)) != -1) bytes.write(buffer, 0, count);
                metadata = new JSONObject(bytes.toString("UTF-8"));
            }
            JSONArray coordinates = metadata.getJSONArray("landmarks");
            float[] landmarks = new float[10];
            for (int i = 0; i < 10; i++) landmarks[i] = (float) coordinates.getDouble(i);
            Bitmap source;
            try (InputStream in = getAssets().open("sample.png")) {
                BitmapFactory.Options options = new BitmapFactory.Options();
                options.inPreferredConfig = Bitmap.Config.ARGB_8888;
                source = BitmapFactory.decodeStream(in, null, options);
            }
            if (source == null) throw new IllegalStateException("Could not decode sample");
            enhancer = new RknnEnhancer();
            JSONObject result = new JSONObject();
            JSONArray timings = new JSONArray();
            Bitmap output = null;
            for (int run = 1; run <= 4; run++) {
                long started = System.nanoTime();
                enhancer.initialize(model); // Cached after the first call.
                Bitmap next = enhancer.enhanceFull(source, landmarks);
                long elapsed = (System.nanoTime()-started)/1_000_000;
                timings.put(elapsed);
                if (output != null) output.recycle();
                output = next;
                Log.i("RknnConsumer", "backend=rknn run="+run+" total_ms="+elapsed);
            }
            result.put("package", getPackageName()).put("backend", "rknn")
                .put("runs_ms", timings).put("width", output.getWidth()).put("height", output.getHeight());
            try (FileOutputStream out = new FileOutputStream(new File(getFilesDir(), "result.png"))) {
                if (!output.compress(Bitmap.CompressFormat.PNG, 100, out)) throw new IllegalStateException("PNG export failed");
            }
            Files.write(new File(getFilesDir(), "result.json").toPath(), result.toString().getBytes(StandardCharsets.UTF_8));
            Log.i("RknnConsumer", "SUCCESS " + result);
            Bitmap displayed = output;
            source.recycle();
            runOnUiThread(() -> { status.setText("RKNN complete. Calls (ms): "+timings); image.setImageBitmap(displayed); });
        } catch (Throwable error) {
            Log.e("RknnConsumer", "FAILED", error);
            runOnUiThread(() -> status.setText(error.toString()));
        }
    }

    @Override public void onDestroy() {
        worker.execute(() -> { if (enhancer != null) enhancer.close(); });
        worker.shutdown();
        super.onDestroy();
    }
}
