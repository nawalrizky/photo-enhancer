package com.enhancer.integrationcheck;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.database.Cursor;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import java.io.File;
import java.io.FileNotFoundException;

/** Demo-only ADB provisioning. Manifest restricts all access to DUMP permission. */
public final class DeploymentProvider extends ContentProvider {
    @Override public boolean onCreate() { return true; }

    @Override public ParcelFileDescriptor openFile(Uri uri, String mode) throws FileNotFoundException {
        File file;
        int flags;
        if ("/model".equals(uri.getPath()) && "w".equals(mode)) {
            File directory = new File(getContext().getFilesDir(), "models");
            if (!directory.isDirectory() && !directory.mkdirs()) throw new FileNotFoundException("Model directory unavailable");
            file = new File(directory, "GFPGANv1.4_rgb_bias_consolidated_rk3576.rknn");
            flags = ParcelFileDescriptor.MODE_WRITE_ONLY | ParcelFileDescriptor.MODE_CREATE | ParcelFileDescriptor.MODE_TRUNCATE;
        } else if (("/result.png".equals(uri.getPath()) || "/result.json".equals(uri.getPath())) && "r".equals(mode)) {
            file = new File(getContext().getFilesDir(), uri.getLastPathSegment());
            flags = ParcelFileDescriptor.MODE_READ_ONLY;
        } else {
            throw new FileNotFoundException("Unsupported deployment URI or mode");
        }
        return ParcelFileDescriptor.open(file, flags);
    }

    @Override public String getType(Uri uri) { return "application/octet-stream"; }
    @Override public Cursor query(Uri u, String[] p, String s, String[] a, String o) { throw new UnsupportedOperationException("File streaming only"); }
    @Override public Uri insert(Uri u, ContentValues v) { throw new UnsupportedOperationException("File streaming only"); }
    @Override public int update(Uri u, ContentValues v, String s, String[] a) { throw new UnsupportedOperationException("File streaming only"); }
    @Override public int delete(Uri u, String s, String[] a) { throw new UnsupportedOperationException("File streaming only"); }
}
